#ifndef SIMULATION_H
#define SIMULATION_H
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#pragma once

#include <omp.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "Data/cereal_help.h"
#include "Data/data_export.h"
#include "Data/logging.h"
#include "Data/param_parser.h"

#include "Mesh/mesh.h"

// Alglib
#include <optimization.h>
#include <stdafx.h>

// Eigen
#include <Eigen/Core>

// Cereal
#include <cereal/types/string.hpp>

class Simulation;

using CsvGetter = std::function<std::string(const Simulation &)>;
struct CsvColumn {
  std::string name;
  CsvGetter getter;
};

class SimulationAlreadyComplete : public std::runtime_error {
public:
  explicit SimulationAlreadyComplete(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief A object used to provide access to data inside the minimization loop
 */
struct DataLink {
  // Gives access to the simulation variables
  Simulation *s;
  // The mesh we are minimizing the energy of
  Mesh *mesh;

  // Alglib has a state.userterminationneeded flag to stop the minimization.
  // Connect this pointer to any stop flag within a minimization algorithm and
  // stop the minimization.
  bool *stopSignal;
  bool *meshWasChanged;

  // The mesh is provided to the minimization function, and sometimes, it is
  // nice to have access to the minimization state. We can get that here
  MinState *minState;
  alglib::minlbfgsstate *LBFGS_state;
  alglib::mincgstate *CG_state;

  // We minimize until all the components of the forces between all the nodes
  // is smaller than this value (Unless another stopping criteria has been
  // reached)
  double *maxForceAllowed;

  // Number of reconnections in the current load step
  int *currentReconnecting;

  DataLink() {};
  DataLink(Simulation *s);
};



// Container for previous load-step and minimization values.
struct SimulationEnergyHistory {
  // Load-step energy tracking (explicit names to avoid ambiguity).
  double prevLoadStepTotalEnergy = 0;
  double prevLoadStepAverageEnergy = 0;
  double loadStepTotalEnergyChange = 0;
  double loadStepAverageEnergyChange = 0;

  // Initial guess values for the current minimization (within a load step).
  double initialGuessTotalEnergy = 0;
  double initialGuessAverageEnergy = 0;
  double totalEnergyChangeFromInitialGuess = 0;
  double averageEnergyChangeFromInitialGuess = 0;
  double initialGuessAverageSigma11 = 0;
  double initialGuessAverageSigma12 = 0;
  double initialGuessAverageSigma22 = 0;
  double initialGuessAverageP11 = 0;
  double initialGuessAverageP12 = 0;
  double initialGuessAverageP21 = 0;
  double initialGuessAverageP22 = 0;
  double averageSigma12ChangeFromInitialGuess = 0;

  // Minimization-iteration energy tracking (for logDuringMinimization).
  double prevMinIterTotalEnergy = 0;
  double prevMinIterAverageEnergy = 0;
  double minIterTotalEnergyChange = 0;
  double minIterAverageEnergyChange = 0;

  template <class Archive> void serialize(Archive &ar) {
    ar(MAKE_NVP(prevLoadStepTotalEnergy), MAKE_NVP(prevLoadStepAverageEnergy),
       MAKE_NVP(loadStepTotalEnergyChange),
       MAKE_NVP(loadStepAverageEnergyChange), MAKE_NVP(initialGuessTotalEnergy),
       MAKE_NVP(initialGuessAverageEnergy),
       MAKE_NVP(totalEnergyChangeFromInitialGuess),
       MAKE_NVP(averageEnergyChangeFromInitialGuess));
    LOAD_WITH_DEFAULT(ar, initialGuessAverageSigma11, 0.0);
    ar(MAKE_NVP(initialGuessAverageSigma12));
    LOAD_WITH_DEFAULT(ar, initialGuessAverageSigma22, 0.0);
    LOAD_WITH_DEFAULT(ar, initialGuessAverageP11, 0.0);
    LOAD_WITH_DEFAULT(ar, initialGuessAverageP12, 0.0);
    LOAD_WITH_DEFAULT(ar, initialGuessAverageP21, 0.0);
    LOAD_WITH_DEFAULT(ar, initialGuessAverageP22, 0.0);
    ar(MAKE_NVP(averageSigma12ChangeFromInitialGuess),
       MAKE_NVP(prevMinIterTotalEnergy), MAKE_NVP(prevMinIterAverageEnergy),
       MAKE_NVP(minIterTotalEnergyChange),
       MAKE_NVP(minIterAverageEnergyChange));
  }
};

/**
 * @brief A object used to controll a loading simulation
 *
 * The object should be initialized with a yaml settings file.
 */
class Simulation {

public:
  // Energy history values tracked per simulation.
  SimulationEnergyHistory energyHistory;

  // Initializes using a config file
  Simulation(Config config, std::string dataPath, bool cleanDataPath = false);
  Simulation() = default;

  // This should run when the mesh is properly pepared. (so we know which nodes
  // are fixed and which nodes are free.)
  void initialize();

  // Sets nrFreeNodes and prepares params and displacement vectors for the
  // minimization algorithms.
  // If some changes are made to the number of fixed nodes mid-simulation, this
  // should be used.
  void initSolver();

  // The first step is special. In order to get the same state across many
  // different settings, we always use the same settings and the same
  // minimization algorithm. That will give simulations (with the same seed) a
  // common stable starting point (as opposed to a common UNSTABLE starting
  // point)
  void firstStep();

  bool keepLoading();

  // Chooses a minimization method and keeps track of minimization time
  // Reconnect is always false if reconnectionMethod is "none"
  MTS_NOINLINE void minimize(bool reconnect = true);

  // Our initial guess will be that all particles have shifted by the same
  // transformation as the border.
  void applyLoadStepToGuess(
      const Matrix2d &guessTransform = Eigen::Matrix2d::Identity());

  // Adds load, applies an affine deformation to the mesh, and updates the
  // initial guess. Uses system deformation for PBC and fixed nodes otherwise.
  void applyAffineStep(const Matrix2d &stepTransform);

  // Runs a forward-backward AQS cycle and tests reversibility (0-1-2-3-4).
  // Returns true if the distance between state 0 and 4 is below eps.
  bool checkReversibility(const Matrix2d &stepTransform, double eps);
  void setReversibilityResult(bool reversible, double distance);
  void addReversibilityCsvColumns();

  // CSV column management (call before initialize() for custom columns).
  template <typename F> void addCsvColumn(std::string name, F getter) {
    addCsvColumnRaw(std::move(name),
                    [g = std::move(getter)](const Simulation &s) {
                      return csvValueToString(g(s));
                    });
  }
  void addDefaultCsvColumns();
  const std::vector<CsvColumn> &getCsvColumns() const { return csvColumns; }

  void addNoiseToGuess(double customNoise = -1);

  MTS_NOINLINE void finishStep(bool reconnect = false);

  // Updates energy history values based on the current mesh state.
  // The mesh should have up-to-date averages before calling this.
  void updateEnergyHistory(bool endOfStep);

  // Computes what proportion of the mesh was involved in a deformation
  void computeParticipationFraction();
  void computeM3ParticipationFraction();

  // Logs a single minimization-state row to the min CSV file.
  MTS_NOINLINE void logMinimizationState();

  // Does some final touches and makes a collection of all the .vtu files in
  // the data folder.
  void finishSimulation();

  // Creates a pvd file that points to all the vtu files in the data folder.
  void gatherDataFiles();

  // Save the simulation to a XML file. Leave fileName empty for default name.
  // Returns the path of the XML file.
  std::string saveSimulation(std::string fileName_ = "");

  // Creates a vtu file of the current state of the simulation
  void writeToFile(bool forceWrite = false, std::string name = "");

  static void loadSimulation(Simulation &s, const std::string &file,
                             const std::string &conf, std::string outputPath,
                             std::optional<bool> forceReRun = std::nullopt);

  // Object used to provide access to various values inside the minimization
  // loop
  DataLink dataLink;

  // The mesh we do our simulations on.
  Mesh mesh;
  std::string reconnectionMethod = "none"; // "none", "edgeFlip", "delaunay"

  // Loading parameters
  double startLoad;
  double loadIncrement;
  double maxLoad;
  // A number from 0 to 1 of loading completion
  double progress;
  // Dimension of mesh
  int rows, cols;

  // participation fraction (Measure of locality in non-affine displacement
  // field)
  double participationFraction;
  double m3ParticipationFraction = 0.0;

  // Folder name
  std::string simName;
  // Path to the output data
  std::string dataPath;

  // Config object
  Config config;

  // Timer to log simulation time
  Timer timer;

  MinState minState;

  // A report to gather information about the minimization
  SimReport FIRERep;
  SimReport LBFGSRep;
  SimReport CGRep;

  // The csv file where we write meta data about each simulation step
  std::ofstream csvFile;

  // The csv file where we write meta data about the internals steps of the
  // minimization algorithm
  std::ofstream minCsvFile;
  std::string minCsvSubfolder;

private:
  // Helper functions
  MTS_NOINLINE void m_minimize(bool rough = false);
  MTS_NOINLINE bool m_reconnect();

  // Uses minlbfgsoptimize to minimize the energy of the system.
  MTS_NOINLINE void m_minimizeWithLBFGS();

  // uses the FIRE algorithm to minimize the energy of the system.
  MTS_NOINLINE void m_minimizeWithFIRE();

  // Uses the conjugate gradient algorithm to minimize the energy of the system.
  MTS_NOINLINE void m_minimizeWithCG();

  void addCsvColumnRaw(std::string name, CsvGetter getter);
  template <typename T> static std::string csvValueToString(const T &v) {
    std::ostringstream oss;
    oss << std::setprecision(11) << v;
    return oss.str();
  }

  bool hasCsvColumn(const std::string &name) const;
  bool tryRecoverCsvColumn(const std::string &name);
  void recoverCsvColumnsFromFile(const std::string &csvPath);

  // Variables alglib uses to give feedback on what happens in the
  // optimization function
  alglib::minlbfgsstate LBFGS_state;
  alglib::minlbfgsreport LBFGS_report;

  alglib::mincgstate CG_state;
  alglib::mincgreport CG_report;

  //change sylvain
  alglib::real_1d_array buildLBFGSScale() const;

  // FIRE parameters
  FIREpp::FIREParam<double> FIRE_param;

  // These values represents the current x and y displacements from the
  // initial position of the simulation
  alglib::real_1d_array alglibNodeDisplacements;
  VectorXd FIRENodeDisplacements;

  // Snapshot before minimization for computing participation fraction
  Mesh::Snapshot beforeMinimization;
  Mesh::Snapshot afterMinimization;
  // Reusable snapshot for reconnection rollback in minimize().
  Mesh::Snapshot reconnectingSnapshot;
  int nrReconnectingCycles = 0;

  struct ReversibilityState {
    int isReversible = 0;
    double distance = 0.0;
    double energyDifference = 0.0;
    double sigma12Difference = 0.0;
    double sigmaTraceDifference = 0.0;
    double sigma11Difference = 0.0;
    double sigma22Difference = 0.0;
    double p11Difference = 0.0;
    double p12Difference = 0.0;
    double p21Difference = 0.0;
    double p22Difference = 0.0;
  };
  ReversibilityState reversibilityState;

  std::vector<CsvColumn> csvColumns;
  bool csvDefaultsAdded = false;
  bool initialized = false;

  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar);

  // Updates the progress (no physics)
  void m_updateProgress();

  // Logs the progress and writes data to disk
  MTS_NOINLINE void m_writeMesh(bool forceWrite = false);
  MTS_NOINLINE void m_writeDump(bool forceWrite = false, std::string name = "");

  // reads the config values to local variables
  void m_loadConfig(Config config);

  // Give DataLink access to private variables
  friend struct DataLink;
  // Test-only access via a helper struct defined in test files.
  friend struct SimulationTestAccess;
};

/**
 * @brief Calculates the energy and forces using the current state of the
 * mesh, pluss a displacement. The first time this function is called, this
 * displacement is the initial guess that we chose.
 *
 * First we update the positions of the mesh, ie, we nudge each node using the
 * values in displacement. Then we calculate the energy and forces for these
 * new positions. Finally, we update the forces so that the minimization
 * function knows how much, and in what direction to nudge the nodes in for
 * the next simulation step.
 *
 * @param mesh A pointer to our mesh
 * @param displacement An array of displacement values for the nodes
 * @param energy The energy of the mesh
 * @param force The force on each node
 */
template <typename ArrayType>
void updateMeshAndComputeForces(DataLink *dataLink, const ArrayType &disp,
                                double &energy, ArrayType &force,
                                int nr_x_values);

// The two following functions use updateMeshAndComputeForces
void alglibEnergyAndGradient(const alglib::real_1d_array &displacement,
                             double &energy, alglib::real_1d_array &force,
                             void *dataLink);
double FIREEnergyAndGradient(Eigen::VectorXd &disp, Eigen::VectorXd &force,
                             void *dataLink);

// Using the nodeDisplacements, we update the position of the nodes
void updateNodePositions(DataLink &dataLink,
                         const alglib::real_1d_array &displacement);
// Overload for Eigen::VectorXd
void updateNodePositions(DataLink &dataLink, const Eigen::VectorXd &disp);

// Updates gradient array from current node forces.
template <typename ArrayType>
void updateGradArray(Mesh *mesh, ArrayType &force, int nr_x_values);

// Creates a simple shear tranformation matrix
Matrix2d getShear(double load, double theta = 0);

// Logs information about non-equilibrium states that occur during minimization.
// Creates a new folder for each loading step during the simulation.
// Warning! Can create a lot of data
void iterationLogger(const alglib::real_1d_array &x, double energy,
                     void *dataLink);

template <class Archive> void Simulation::serialize(Archive &ar) {
  ar(MAKE_NVP(rows), MAKE_NVP(cols), MAKE_NVP(mesh), MAKE_NVP(dataPath),
     MAKE_NVP(timer), MAKE_NVP(simName), MAKE_NVP(config));
  LOAD_WITH_DEFAULT(ar, energyHistory, SimulationEnergyHistory{});
}

#endif
