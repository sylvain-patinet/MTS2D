#include "simulation.h"
#include "Data/cereal_help.h"
#include "Data/data_export.h"
#include "Data/logging.h"
#include "Data/param_parser.h"
#include "Mesh/node.h"
#include "randomUtils.h"
#include "settings.h"
#include <Eigen/LU>
#include <FIRE.h>
#include <Param.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <optimization.h>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

Simulation::Simulation(Config config_, std::string _dataPath,
                       bool cleanDataPath) {
  dataPath = _dataPath;

  // This function initializes a lot of the variables using the config file
  m_loadConfig(config_);

  timer = Timer();

  mesh = Mesh(rows, cols, 1, config.QDSD, config.usingPBC, config.meshDiagonal,
              config.energyFunction, config.bulkModulus);
  mesh.load = startLoad;
  mesh.setSimNameAndDataPath(simName, dataPath);
  addDefaultCsvColumns();

  if (simulationAlreadyComplete(simName, dataPath, maxLoad) &&
      !config.forceReRun) {
    throw SimulationAlreadyComplete("Simulation already complete");
  }
  if (cleanDataPath) {

    clearOutputFolder(simName, dataPath);
    saveConfigFile(config, dataPath);
    // Create and open file
  }
}

void Simulation::initialize() {
  if (initialized) {
    return;
  }
  timer.addKey("minimization");
  // Initialization should be done after nodes have been moved and fixed as
  // desired. The elements created by the function below are copies and do not
  // dynamically update. (the update function only updates the position,
  // energy and stress)

  // initialization of the csv file needs to be done after the correct loadStep
  // has been loaded
  // TODO: Keeping the csv file open during the entire simulation might be
  // unwise. We should perhaps save batches of lines, and open and close the
  // file as needed.
  addDefaultCsvColumns();
  csvFile = initCsvFile(simName, dataPath, *this);

  // We assume that the nodes already contain information about the mesh
  // structure, therefore, we recreate the elements
  // mesh.recreateElements();

  // we update the solvers
  initSolver();

  // Start simulation timer
  timer.Start();
  initialized = true;
}

void Simulation::initSolver() {

  // Alglib Initialization preparation
  int nrFreeNodes = mesh.freeNodeIds.size();
  alglibNodeDisplacements.setlength(2 * nrFreeNodes);
  // Set values to zero
  for (int i = 0; i < 2 * nrFreeNodes; i++) {
    alglibNodeDisplacements[i] = 0;
  }

  // Adjust nrCorrections based on the number of free nodes
  // alglib doesn't like that nr corrections is larger than nr of free nodes
  if (config.LBFGSNrCorrections > 2 * nrFreeNodes) {
    config.LBFGSNrCorrections = 2 * nrFreeNodes;
  }

  if (nrFreeNodes != 0) {
    // https://www.alglib.net/translator/man/manual.cpp.html#sub_minlbfgscreate
    // Initialize the state variable
    alglib::minlbfgscreate(config.LBFGSNrCorrections, alglibNodeDisplacements,
                           LBFGS_state);

    alglib::mincgcreate(alglibNodeDisplacements, CG_state);
  }

  // FIRE Initialization
  FIRENodeDisplacements = VectorXd::Zero(2 * nrFreeNodes);
  FIRE_param = FIREpp::FIREParam<double>(mesh.nrNodes, 2);
  config.updateParam(FIRE_param);

  reconnectingSnapshot.displacements.resize(2 * mesh.nrNodes);

  // We update the datalink with the latest information about out simulation
  // data
  dataLink = DataLink(this);
  // TEMP
  // config.logDuringMinimization = true;
}

void Simulation::firstStep() {
  if (!initialized) {
    initialize();
  }

  if (mesh.loadSteps == 0 && startLoad != 0.0) {
    Matrix2d startLoadTransform = getShear(startLoad);
    mesh.applyTransformation(startLoadTransform);
  }

  // If it is the first step, we always minimize with the same algorithm to
  // ensure each seed has the same STABLE starting point
  // And we'll change the settings so that we get the same starting point for
  // all simulations
  double oldMaxForceAllowed = *dataLink.maxForceAllowed;
  double newMaxForceAllowed = 1e-6;
  *dataLink.maxForceAllowed = newMaxForceAllowed;

  std::string oldMinimizer = config.minimizer;
  std::string newMinimizer = "LBFGS";
  config.minimizer = newMinimizer;

  // Pretend to add load to correctly count load steps
  mesh.addLoad(0);
  // Set initial guess to the current position (starting load)
  applyLoadStepToGuess();
  // Add noise (from a seed) to trigger an avalanche
  addNoiseToGuess();
  // Minimizes the energy by moving the free nodes in the mesh
  minimize();
  // Updates progress and writes to file
  finishStep();

  // Reset settings
  *dataLink.maxForceAllowed = oldMaxForceAllowed; // Restore original value
  config.minimizer = oldMinimizer;

  // Reset LBFGS report
  // Usually done before minimization, but will not be done if we are not using
  // LBFGS for the other loading steps
  LBFGS_report = alglib::minlbfgsreport();
  LBFGSRep = SimReport(LBFGS_report);
}

bool Simulation::keepLoading() {
  double nextLoad = mesh.load + loadIncrement;
  double buffer = 0.5 * loadIncrement;

  if (loadIncrement > 0) {
    return nextLoad <= maxLoad + buffer;
  } else {
    return nextLoad >= startLoad + buffer;
  }
}

// Helper function
void Simulation::m_minimize(bool rough) {
  if (rough) {
    double roughMaxForceAllowed = 1e-2;
    dataLink.maxForceAllowed = &roughMaxForceAllowed;
  }

  auto fail = [&](const std::string &msg) -> void {
    std::cerr << msg << '\n';
    writeToFile(true, "CrashAtLoad:" + std::to_string(mesh.load));
    throw std::runtime_error("Minimization failed: " + msg);
  };

  try {
    if (config.minimizer == "FIRE") {
      m_minimizeWithFIRE();
      mesh.nrMinItterations += FIRERep.nrIter;
    } else if (config.minimizer == "LBFGS") {
      m_minimizeWithLBFGS();
      mesh.nrMinItterations += LBFGSRep.nrIter;
    } else if (config.minimizer == "CG") {
      m_minimizeWithCG();
      mesh.nrMinItterations += CGRep.nrIter;
    } else {
      throw std::invalid_argument("Unknown minimizer: " + config.minimizer);
    }
  } catch (const alglib::ap_error &e) {
    fail("ALGLIB error: " + std::string(e.msg));
  } catch (const std::exception &e) {
    fail("Standard exception: " + std::string(e.what()));
  } catch (...) {
    fail("Unknown exception caught!");
  }
  // if (FIRERep.termType == -3) {
  //  writeToFile(true);
  //  throw std::runtime_error("Energy too high");
  //}
  if (LBFGSRep.termType == 1) {
    // mesh.writeToVtu("badStopStep" + std::to_string(mesh.loadSteps));
    // If maxForceAllowed is smaller than 1e-19, we assume it is zero and not
    // used
    if (mesh.maxForce > 1.5 * (*dataLink.maxForceAllowed) &&
        *dataLink.maxForceAllowed > 1e-19) {
      if (!isQuiet()) {
        std::cout << "Warning: LBFGS stopped with termType 1 but max force is "
                     "still high: "
                  << mesh.maxForce << '\n';
      }
    }
  }
  if (rough) {
    dataLink.maxForceAllowed = &config.epsR;
  }
}

bool Simulation::m_reconnect() {
  // std::cout << "Reconnecting\n";
  if (reconnectionMethod == "edgeFlip") {
    return mesh.reconnect();
  } else if (reconnectionMethod == "delaunay") {
    mesh.reconnectDelaunay();
    return true; // TODO check if Delaunay made any changes
  } else if (reconnectionMethod == "none") {
    // Do nothing
  } else {
    throw std::invalid_argument("Unknown reconnection method: " +
                                reconnectionMethod);
  }
  return true;
}

void Simulation::minimize(bool reconnect) {
  /*
  Pseudo code for minimization with reconnection:

  testEnergy = minimize()
  do:
    saveState()
    bestEnergy = testEnergy
    reconnect()
    testEnergy = minimize()
  while (testEnergy < bestEnergy)

  Revert to saved state.
  */
  if (mesh.freeNodeIds.size() == 0) {
    return;
  }
  if (reconnectionMethod == "none") {
    reconnect = false; // We don't need to run the minimization multiple times
                       // if we don't reconnect
  }
  timer.Start("minimization");
  minCsvSubfolder.clear();

  // If we log during minimization, we need a new file for each minimization
  if (config.logDuringMinimization) {
    minCsvSubfolder =
        std::string(DATAFOLDERPATH) + "/" + getMinDataSubFolder(mesh);
    minCsvFile = initCsvFile(simName, dataPath, *this, minCsvSubfolder);
  }

  // Save mesh before minimization for calculation of paticipation fraction
  mesh.saveSnapshot(beforeMinimization);

  // First minimization (If we reconnect, we also run a rough minimization)
  m_minimize();

  // std::cout << "Step: " << mesh.loadSteps << '\n';

  // We only reconnect if there is an energy drop since last load step
  // if (mesh.totalEnergy > energyHistory.prevLoadStepTotalEnergy) {
  //   reconnect = false;
  // }
  if (mesh.nr_elements_with_m3_fix_changeInStep == 0) {
    reconnect = false;
  }

  if (!reconnect) { // If we don't reconnect, we are done after one
    // minimization
    logMinimizationState();
    // Save mesh after minimization for calculation of participation fraction
    mesh.saveSnapshot(afterMinimization);
    timer.Stop("minimization");
    return;
  }

  nrReconnectingCycles = 0;
  double testEnergy; // These help readability. They could be replaced
  double bestEnergy; // with mesh.totalEnergy and a cached snapshot
  bool meshChanged;  // We keep track of whether the mesh was changed by the
                     // reconnection, to avoid unnecessary minimizations. If the
                     // mesh didn't change, the energy won't change either.
  testEnergy = mesh.totalEnergy;
  do {
    bestEnergy = testEnergy; // We only repeat if testEnergy < bestEnergy
    mesh.saveSnapshot(reconnectingSnapshot);

    nrReconnectingCycles++;
    if (nrReconnectingCycles % 10 == 0) {
      std::cout << "Step: " << mesh.loadSteps
                << " Reconnections: " << nrReconnectingCycles << std::endl;
    }

    if (config.logDuringMinimization) {
      mesh.writeToVtu("", true, VtuFieldLevel::All, "pre");
    }
    meshChanged = m_reconnect();
    if (config.logDuringMinimization) {
      mesh.writeToVtu("", true, VtuFieldLevel::All, "post");
    }
    if (!meshChanged) {
      break;
    }
    m_minimize();
    testEnergy = mesh.totalEnergy;

    // Reconnection reduced the energy. Perhaps we can reduce further with
    // another reconnection.
  } while (testEnergy < bestEnergy);

  // If we got here AND the mesh changed, it means the last reconnection made
  // things worse, so we revert to the previous state
  if (meshChanged) {
    if (reconnectionMethod == "edgeFlip") {
      mesh.undoReconnections();
    }
    // revert also calculates energy and forces
    mesh.restoreState(reconnectingSnapshot);
    const double restoredEnergy = mesh.totalEnergy;
    const double energyDiff = std::abs(restoredEnergy - bestEnergy);
    const double energyTol =
        1e-10 * std::max(1.0, std::abs(bestEnergy));
    if (energyDiff > energyTol) {
      std::ostringstream msg;
      msg << "Reconnection revert energy mismatch: expected " << bestEnergy
          << ", got " << restoredEnergy << " (diff " << energyDiff << ").";
      throw std::runtime_error(msg.str());
    }
  }
  logMinimizationState();

  // Save mesh after minimization for calculation of paticipation fraction
  mesh.saveSnapshot(afterMinimization);

  timer.Stop("minimization");
}

//change sylvain
alglib::real_1d_array Simulation::buildLBFGSScale() const {
  const int nrFreeNodes = mesh.freeNodeIds.size();
  alglib::real_1d_array s;
  s.setlength(2 * nrFreeNodes);

  // First test: neutral scaling
  // Later you can replace 1.0 by a characteristic displacement scale.
  for (int i = 0; i < 2 * nrFreeNodes; ++i) {
    s[i] = 1.0;
  }

  return s;
}
//change sylvain

//change sylvain
// void Simulation::m_minimizeWithLBFGS() {
//   timer.Start("LBFGSMinimization");
//   // https://www.alglib.net/translator/man/manual.cpp.html#sub_minlbfgsrestartfrom
//   // We reset and reuse the state instead of initializing it again
//   // (The hessian is reset and not preserved)
//   alglib::minlbfgsrestartfrom(LBFGS_state, alglibNodeDisplacements);
//
//   // Set termination condition, ei. when is the solution good enough
//   // https://www.alglib.net/translator/man/manual.cpp.html#sub_minlbfgssetcond
//   alglib::minlbfgssetcond(LBFGS_state, config.LBFGSEpsg, config.LBFGSEpsf,
//                           config.LBFGSEpsx, config.LBFGSMaxIterations);
//
//   // Connect the chosen state to the minimization state
//   minState = MinState(LBFGS_state);
//
//   //  This is where the heavy calculations happen
//   alglib::minlbfgsoptimize(LBFGS_state, alglibEnergyAndGradient,
//                            iterationLogger, &dataLink);
//
//   alglib::minlbfgsresults(LBFGS_state, alglibNodeDisplacements, LBFGS_report);
//   LBFGSRep.nms = timer.Stop("LBFGSMinimization");
//   LBFGSRep = SimReport(LBFGS_report);
// }


void Simulation::m_minimizeWithLBFGS() {
  timer.Start("LBFGSMinimization");

  // Reuse optimizer state but restart from current point.
  alglib::minlbfgsrestartfrom(LBFGS_state, alglibNodeDisplacements);

  // --- Change 1: variable scaling ---
  alglib::real_1d_array lbfgsScale = buildLBFGSScale();
  alglib::minlbfgssetscale(LBFGS_state, lbfgsScale);

  // --- Change 2: diagonal scale-based preconditioning ---
  // This uses the scale vector above as a diagonal preconditioner.
  alglib::minlbfgssetprecscale(LBFGS_state);

  // Stopping conditions
  alglib::minlbfgssetcond(
      LBFGS_state,
      config.LBFGSEpsg,
      config.LBFGSEpsf,
      config.LBFGSEpsx,
      config.LBFGSMaxIterations
  );

  // Connect the chosen state to the minimization state
  minState = MinState(LBFGS_state);

  // Heavy calculations happen here
  alglib::minlbfgsoptimize(
      LBFGS_state,
      alglibEnergyAndGradient,
      iterationLogger,
      &dataLink
  );

  alglib::minlbfgsresults(LBFGS_state, alglibNodeDisplacements, LBFGS_report);

  LBFGSRep.nms = timer.Stop("LBFGSMinimization");
  LBFGSRep = SimReport(LBFGS_report);
}
//change sylvain



void Simulation::m_minimizeWithCG() {
  timer.Start("CGMinimization");
  // https://www.alglib.net/translator/man/manual.cpp.html#sub_mincgrestartfrom
  // We reset and reuse the state instead of initializing it again
  alglib::mincgrestartfrom(CG_state, alglibNodeDisplacements);

  // Set termination condition, ei. when is the solution good enough
  // https://www.alglib.net/translator/man/manual.cpp.html#sub_mincgsetcond
  alglib::mincgsetcond(CG_state, config.CGEpsg, config.CGEpsf, config.CGEpsx,
                       config.CGMaxIterations);

  // TODO!
  // minState = MinState(CG_state);

  // This is where the heavy calculations happen
  // The null pointer can be replaced with a logging function
  // https://www.alglib.net/translator/man/manual.cpp.html#sub_mincgoptimize
  alglib::mincgoptimize(CG_state, alglibEnergyAndGradient, iterationLogger,
                        &dataLink);

  alglib::mincgresults(CG_state, alglibNodeDisplacements, CG_report);
  CGRep.nms = timer.Stop("CGMinimization");
  CGRep = SimReport(CG_report);
}

template <typename ArrayType>
void updateMeshAndComputeForces(DataLink *dataLink, const ArrayType &disp,
                                double &energy, ArrayType &grad,
                                int nr_x_values) {
  Mesh *mesh = dataLink->mesh;

  // Update mesh position from the result of the
  // previous minimization
  updateNodePositions(*mesh, disp);

  // Calculate energy and forces
  mesh->updateMesh();
  assert(mesh->getUpdateState() == Mesh::UpdateState::Forces);

  // Total energy, only used for minimization
  energy = mesh->totalEnergy;

  // Update gradient in the minimization
  updateGradArray(mesh, grad, nr_x_values);
  double maxForce = mesh->maxForce;
  // int it = dataLink->LBFGS_state->c_ptr()->repiterationscount;

  // Determine if the minimization is done
  if (maxForce < *dataLink->maxForceAllowed) {
    // stop the minimization
    // Check if the current step is a valid step
    // TODO: Even if the max force is small right now, it might become large
    // when lbfgs selects the step with the lowest energy.
    alglib::minlbfgsrequesttermination(*dataLink->LBFGS_state);
    alglib::mincgrequesttermination(*dataLink->CG_state);
  }
  // We start to reconnect once we are 'close' to a solution
  // And only reconnect every 100 iterations
  // else if (mesh->nr_elements_with_m3_fix_changeInStep > 0) {

  //   mesh->reconnect(false, true);
  //   if (mesh->reconnectRequired) {
  //     // stop the minimization
  //     alglib::minlbfgsrequesttermination(*dataLink->LBFGS_state);
  //     alglib::mincgrequesttermination(*dataLink->CG_state);
  //   }
  // }
  // Since we don't use the x displacement argument in the iteration logger,
  // we just pass default parameter
  iterationLogger(alglib::real_1d_array(), energy, dataLink);
  // Update pastPlastic count so we can save files when it changes
  // mesh->resetPastPlasticCount(false);
  if (mesh->nrMinFunctionCalls == 0 &&
      !dataLink->s->config.logDuringMinimization) {
    // Ensure initial-guess stats are computed.
    // If we log during minimization, these will be updated in the iteration
    // logger instead
    mesh->updateAveragesAndPlasticEvents();
    dataLink->s->updateEnergyHistory(false);
  }
  mesh->nrMinFunctionCalls++;
}

void alglibEnergyAndGradient(const alglib::real_1d_array &disp, double &energy,
                             alglib::real_1d_array &grad, void *dataLinkPtr) {
  DataLink *dataLink = reinterpret_cast<DataLink *>(dataLinkPtr);
  updateMeshAndComputeForces(dataLink, disp, energy, grad, grad.length() / 2);
}

double FIREEnergyAndGradient(Eigen::VectorXd &disp, Eigen::VectorXd &grad,
                             void *dataLinkPtr) {
  double energy;
  DataLink *dataLink = reinterpret_cast<DataLink *>(dataLinkPtr);
  updateMeshAndComputeForces(dataLink, disp, energy, grad, grad.size() / 2);
  return energy;
}

void Simulation::m_minimizeWithFIRE() {
  timer.Start("FIREMinimization");
  FIREpp::FIRESolver<double> s = FIREpp::FIRESolver<double>(FIRE_param);
  double energy;
  FIRERep.nrIter = s.minimize(FIREEnergyAndGradient, FIRENodeDisplacements,
                              energy, &dataLink, FIRERep.termType);
  FIRERep.nms = timer.Stop("FIREMinimization");
  FIRERep.nfev = mesh.nrMinFunctionCalls;

  if (FIRERep.termType == -3) {
    writeToFile(true);
  }
  // We first do FIRE, but then use the result as an initial guess for LBFGS
  // setInitialGuess();
  // m_minimizeWithLBFGS();
}

// Overload for alglib::real_1d_array
void updateNodePositions(Mesh &mesh, const alglib::real_1d_array &disp) {
  mesh.updateNodePositions(disp.getcontent(), disp.length());
}

// Overload for Eigen::VectorXd
void updateNodePositions(Mesh &mesh, const Eigen::VectorXd &disp) {
  mesh.updateNodePositions(disp.data(), disp.size());
}

// Updates gradient array from current node forces.
template <typename ArrayType>
void updateGradArray(Mesh *mesh, ArrayType &grad, int nr_x_values) {
  for (int i = 0; i < nr_x_values; i++) {
    NodeId n_id = mesh->freeNodeIds[i];
    const auto &node = (*mesh)[n_id];

    // Store force values in the array
    double fx = node->f[0];
    double fy = node->f[1];

    // Gadient is the oposite sign of the force
    grad[i] = -fx;
    grad[nr_x_values + i] = -fy;
  }
}

void Simulation::applyLoadStepToGuess(const Matrix2d &T) {
  // NOTE: T is expected to be the *incremental* deformation since the last
  // guess. This function composes T with the current displacements, so calling
  // it twice with the same T applies the deformation twice.
  const int nr_x_values = alglibNodeDisplacements.length() / 2;
  const Matrix2d I = Matrix2d::Identity();
  const Matrix2d A = T - I; // often small

  for (size_t i = 0; i < mesh.freeNodeIds.size(); i++) {
    const Node *n = mesh[mesh.freeNodeIds[i]];

    // u = pos - init_pos (small)
    const Vector2d u = n->u();

    // stable form: (T - I)*init_pos + T*u
    // Explination: u2 = Tx-x0, x = x0 + u1
    // Insert: u2 = T(x0+u1)-x0
    // Collect x0: u2 = (T-I)x0+Tu1
    const Vector2d next_displacement = A * n->ref_pos() + T * u;

    alglibNodeDisplacements[i] = next_displacement.x();
    alglibNodeDisplacements[i + nr_x_values] = next_displacement.y();
    FIRENodeDisplacements[i] = next_displacement.x();
    FIRENodeDisplacements[i + nr_x_values] = next_displacement.y();
  }

  // Not strictly necessary for minimization (the solver applies displacements
  // internally), but keeping mesh positions in sync avoids confusion. If you
  // rebuild a guess by calling setInitialGuess again, the mesh displacements
  // are used as the baseline.
  updateNodePositions(mesh, alglibNodeDisplacements);
}

void Simulation::applyAffineStep(const Matrix2d &T) {
  mesh.addLoad(loadIncrement);
  if (mesh.usingPBC) {
    mesh.applyTransformationToSystemDeformation(T);
  } else {
    mesh.applyTransformationToFixedNodes(T);
  }
  applyLoadStepToGuess(T);
}

void Simulation::setReversibilityResult(bool reversible, double distance) {
  reversibilityState.isReversible = reversible ? 1 : 0;
  reversibilityState.distance = distance;
}

void Simulation::addCsvColumnRaw(std::string name, CsvGetter getter) {
  csvColumns.push_back({std::move(name), std::move(getter)});
}

bool Simulation::hasCsvColumn(const std::string &name) const {
  for (const auto &col : csvColumns) {
    if (col.name == name) {
      return true;
    }
  }
  return false;
}

bool Simulation::tryRecoverCsvColumn(const std::string &name) {
  if (name == "is_reversible" || name == "rev_d") {
    addReversibilityCsvColumns();
    return true;
  }
  return false;
}

static std::vector<std::string> splitCsvHeaderLine(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

void Simulation::recoverCsvColumnsFromFile(const std::string &csvPath) {
  std::ifstream file(csvPath);
  if (!file.is_open()) {
    return;
  }
  std::string header;
  if (!std::getline(file, header)) {
    return;
  }
  const auto headers = splitCsvHeaderLine(header);
  for (const auto &name : headers) {
    if (!hasCsvColumn(name)) {
      tryRecoverCsvColumn(name);
    }
  }
}

// Default CSV columns. Use (expr) to keep commas inside a single macro arg.
#define DEFAULT_CSV_COLS(X)                                                    \
  X("load_step", s.mesh.loadSteps)                                             \
  X("load", s.mesh.load)                                                       \
  X("total_energy", s.mesh.totalEnergy)                                        \
  X("total_energy_change", s.energyHistory.loadStepTotalEnergyChange)          \
  X("total_init_energy", s.energyHistory.initialGuessTotalEnergy)              \
  X("total_e_change_from_init",                                                \
    s.energyHistory.totalEnergyChangeFromInitialGuess)                         \
  X("avg_energy", s.mesh.averageEnergy)                                        \
  X("avg_energy_change", s.energyHistory.loadStepAverageEnergyChange)          \
  X("avg_init_energy", s.energyHistory.initialGuessAverageEnergy)              \
  X("avg_e_change_from_init",                                                  \
    s.energyHistory.averageEnergyChangeFromInitialGuess)                       \
  X("min_iter_total_energy_change", s.energyHistory.minIterTotalEnergyChange)  \
  X("min_iter_avg_energy_change", s.energyHistory.minIterAverageEnergyChange)  \
  X("max_energy", s.mesh.maxEnergy)                                            \
  X("max_force", s.mesh.maxForce)                                              \
  X("avg_sigma11", s.mesh.averageSigma11)                                      \
  X("avg_sigma12", s.mesh.averageSigma12)                                      \
  X("avg_sigma22", s.mesh.averageSigma22)                                      \
  X("avg_init_sigma11", s.energyHistory.initialGuessAverageSigma11)            \
  X("avg_init_sigma12", s.energyHistory.initialGuessAverageSigma12)            \
  X("avg_init_sigma22", s.energyHistory.initialGuessAverageSigma22)            \
  X("avg_sigma12_change_from_init",                                            \
    s.energyHistory.averageSigma12ChangeFromInitialGuess)                      \
  X("avg_P11", s.mesh.averageP11)                                              \
  X("avg_P12", s.mesh.averageP12)                                              \
  X("avg_P21", s.mesh.averageP21)                                              \
  X("avg_P22", s.mesh.averageP22)                                              \
  X("avg_init_P11", s.energyHistory.initialGuessAverageP11)                    \
  X("avg_init_P12", s.energyHistory.initialGuessAverageP12)                    \
  X("avg_init_P21", s.energyHistory.initialGuessAverageP21)                    \
  X("avg_init_P22", s.energyHistory.initialGuessAverageP22)                    \
  X("participationFraction", s.participationFraction)                          \
  X("m3_participationFraction", s.m3ParticipationFraction)                     \
  X("nr_elements_with_m3_fix_change", s.mesh.nr_elements_with_m3_fix_change)   \
  X("nr_red_q1", s.mesh.redQuadrantCounts[0])                                  \
  X("nr_red_q2", s.mesh.redQuadrantCounts[1])                                  \
  X("nr_red_q3", s.mesh.redQuadrantCounts[2])                                  \
  X("nr_red_q4", s.mesh.redQuadrantCounts[3])                                  \
  X("nr_red_q1_fixed", s.mesh.redQuadrantFixedCounts[0])                       \
  X("nr_red_q2_fixed", s.mesh.redQuadrantFixedCounts[1])                       \
  X("nr_red_q3_fixed", s.mesh.redQuadrantFixedCounts[2])                       \
  X("nr_red_q4_fixed", s.mesh.redQuadrantFixedCounts[3])                       \
  X("max_m3_nr", s.mesh.maxM3Nr)                                               \
  X("sum_m3", s.mesh.sumM3Nr)                                                  \
  X("max_positive_plastic_jump", s.mesh.maxPlasticJump)                        \
  X("max_negative_plastic_jump", s.mesh.minPlasticJump)                        \
  X("nr_iterations", s.mesh.nrMinItterations)                                  \
  X("nr_func_evals", s.mesh.nrMinFunctionCalls)                                \
  X("nr_edge_flips", s.mesh.edgeFlipsSinceLastStep)                            \
  X("LBFGS_Term_reason", s.LBFGSRep.termType)                                  \
  X("CG_Term_reason", s.CGRep.termType)                                        \
  X("FIRE_Term_reason", s.FIRERep.termType)                                    \
  X("run_time", (s.timer.RTString()))                                          \
  X("minimization_time", (s.timer.RTString("minimization", 7)))                \
  X("write_time", (s.timer.RTString("write", 7)))                              \
  X("est_time_remaining", s.timer.oldETRString)                                \
  X("cmX", s.mesh.com[0])                                                      \
  X("cmY", s.mesh.com[1])                                                      \
  X("maxX", s.mesh.bounds[0])                                                  \
  X("minX", s.mesh.bounds[1])                                                  \
  X("maxY", s.mesh.bounds[2])                                                  \
  X("minY", s.mesh.bounds[3])

void Simulation::addDefaultCsvColumns() {
  if (csvDefaultsAdded) {
    return;
  }
  csvDefaultsAdded = true;
// If you are trying to understand how to add columns, look at:
// addCsvColumn(name, [](const auto &s) { return (expr); })
// This can be used to add a column.
#define ADD_COL(name, expr)                                                    \
  addCsvColumn(name, [](const auto &s) { return (expr); });
  DEFAULT_CSV_COLS(ADD_COL)
#undef ADD_COL
}

#undef DEFAULT_CSV_COLS

// Core function to add (gausian) noise to a double array
void addNoiseToArray(double *data, size_t length, double noise) {
  double dataSum = 0;
  for (size_t i = 0; i < length; i++) {
    // Generate random noise from a normal distribution with mean 0 and
    // stddev noise
    data[i] += sampleNormal(0.0, noise);
    dataSum += data[i];
  }
  // Print a noise hash to easily confirm two identical simulations
  // use as many decimal places as possible
  if (!isQuiet()) {
    std::cout << "Noise hash: " << std::scientific << std::setprecision(15)
              << dataSum << std::endl;
  }
}

// Overload for alglib::real_1d_array
void addNoise(alglib::real_1d_array &array, double noise) {
  addNoiseToArray(array.getcontent(), array.length(), noise);
}

// Overload for Eigen::VectorXd
void addNoise(Eigen::VectorXd &vector, double noise) {
  addNoiseToArray(vector.data(), vector.size(), noise);
}

// adds noise to both alglib and fire guess
void Simulation::addNoiseToGuess(double customNoise) {
  // Choose whether or not to use noise from argument or config file
  double effectiveNoise =
      customNoise == -1 ? config.initialGuessNoise : customNoise;

  if (config.minimizer == "FIRE") {
    addNoise(FIRENodeDisplacements, effectiveNoise);
  } else {
    addNoise(alglibNodeDisplacements, effectiveNoise);
  }
  // Noise only updates the displacement arrays. If you need the mesh node
  // positions to reflect the noisy guess (e.g., before saving a mesh copy),
  // call updateNodePositions(mesh,
  // alglibNodeDisplacements/FIRENodeDisplacements) explicitly.
}

void Simulation::m_updateProgress() {
  using namespace std::chrono;

  progress =
      std::clamp((mesh.load - startLoad) / (maxLoad - startLoad), 0.0, 1.0);

  int intProgress = static_cast<int>(progress * 100);

  std::string consoleProgressMessage =
      std::to_string(intProgress) + "%"         //
      + " RT: " + timer.RTString()              //
      + "\tETR: " + timer.ETRString(progress)   //
      + "\tLoad: " + std::to_string(mesh.load); //

  // Use static variables to track the last progress and the last update
  // time
  static int oldProgress = -1;
  static int firstProgress = -1;
  static int lastDump = -1;
  static auto lastUpdateTime = steady_clock::now();

  auto now = steady_clock::now();
  auto timeSinceLastUpdate =
      duration_cast<seconds>(now - lastUpdateTime).count();

  bool shouldUpdate =
      !isQuiet() && (oldProgress != intProgress || timeSinceLastUpdate >= 20);

  if (shouldUpdate) {
    oldProgress = intProgress;
    firstProgress = (firstProgress == -1) ? intProgress : firstProgress;
    lastUpdateTime = now;

    std::cout << consoleProgressMessage << std::endl;

    if (!csvFile.is_open()) {
      throw std::runtime_error("CSV file stream is not open.");
    }
    csvFile.flush(); // Update the file on disk

    // Check if we should make a dump
    if (intProgress % 5 == 0 && intProgress != lastDump &&
        firstProgress != intProgress) {
      lastDump = intProgress;
      m_writeDump(true);
    }
  }
}

void Simulation::writeToFile(bool forceWrite, std::string fileName) {
  timer.Start("write");
  // We write to the cvs file every time this function is called
  writeToCsv(csvFile, (*this));
  // These are writing date much less often
  m_writeMesh(forceWrite);
  m_writeDump(forceWrite, fileName);
  if (config.logDuringMinimization) {
    // If we are logging minimization, we want to keep the folder only for
    // steps with enough plasticity or energy drop.
    const bool hasPlasticEvents =
        mesh.nr_elements_with_m3_fix_change >
        mesh.nrElements * config.plasticityEventThreshold;
    const bool hasEnergyDrop =
        -energyHistory.totalEnergyChangeFromInitialGuess >
        config.energyDropThreshold;
    const bool periodicKeep = (mesh.loadSteps % 1000 == 0);
    const bool keepMinFolder =
        hasPlasticEvents || hasEnergyDrop || periodicKeep;

    if (minCsvSubfolder.empty()) {
      timer.Stop("write");
      return;
    }

    const std::string minFolderPath =
        getOutputPath(simName, dataPath) + "/" + minCsvSubfolder;

    if (keepMinFolder) {
      // Create a collection after each step we keep.
      createCollection(
          minFolderPath, minFolderPath,
          "^.*_minStep=[0-9]+\\.([0-9]+)(?:_[^.]+)?\\.[0-9]+\\.vtu$");
    } else {
      namespace fs = std::filesystem;
      fs::path folderPath(minFolderPath);
      if (fs::exists(folderPath) && fs::is_directory(folderPath)) {
        if (minCsvFile.is_open()) {
          minCsvFile.close();
        }
        fs::remove_all(folderPath);
        minCsvSubfolder.clear();
      }
    }
  }
  timer.Stop("write");
}

void Simulation::m_writeMesh(bool forceWrite) {
  // Only if there are lots of plastic events will we want to save the data.
  // If we save every frame, it requires too much storage.
  // (A 100x100 system loaded from 0.15 to 1 with steps of 1e-5 would take
  // up 180GB) At the same time, if there are few large avalanvhes, we might
  // go long without saving data. In order to get a good framerate for an
  // animation, we want to ensure that not too much happens between frames.
  // The following enures that we at least have 200 frames of states over
  // the course of loading, but also don't miss any big events
  static double lastLoadWritten = 0;
  if ((mesh.nr_elements_with_m3_fix_change >
       mesh.nrElements *
           config.plasticityEventThreshold) || // Lots of plastic change
      (-energyHistory.totalEnergyChangeFromInitialGuess >
       config.energyDropThreshold) ||               // Large energy drop
      (abs(mesh.load - lastLoadWritten) > 0.005) || // Absolute change
      (abs(mesh.load - lastLoadWritten) / (maxLoad - startLoad) >
       0.005) ||  // Relative change
      forceWrite) // Force write
  {
    mesh.writeToVtu();
    lastLoadWritten = mesh.load;
  }
}

void Simulation::m_writeDump(bool forceWrite, std::string name) {
  if (!config.writeDumps) {
    return;
  }
  // When do we create save states?
  // I'm thinking I want to do one halfway no matter how short the
  // simulation is, but then outside of that, i'm thinking once per hour is
  // okay.
  auto now = std::chrono::steady_clock::now();
  static auto lastSaveTime =
      now; // Since this is a static variable, this line is only run once
  static bool firstSaveDone = false;
  static int lastDefaultDumpLoadStep = -1;
  auto elapsedSinceLastSave =
      std::chrono::duration_cast<std::chrono::seconds>(now - lastSaveTime);
  double midPointLoad = (startLoad + maxLoad) / 2;

  // Save every one hours
  static const std::chrono::hours saveFrequency(1);

  bool shouldSave =
      (mesh.load >= midPointLoad &&
       !firstSaveDone) || // Check for first save at midpoint
      (elapsedSinceLastSave >= saveFrequency) || // Hourly save
      mesh.load + loadIncrement / 2 > maxLoad || // Check for final save
      forceWrite;                                // Check if forced

  if (shouldSave) {
    if (name.empty() && lastDefaultDumpLoadStep == mesh.loadSteps) {
      return;
    }
    saveSimulation(name);
    if (name.empty()) {
      lastDefaultDumpLoadStep = mesh.loadSteps;
    }

    // Perhaps a bit strange, but this seems like a nice time to also
    // create/update the pvd file. (Sometimes it can be nice to have this
    // file before the simulation is done)
    gatherDataFiles();
    lastSaveTime = now;
    firstSaveDone = true;
  }
}

void Simulation::finishStep(bool reconnect) {
  // if (reconnect) {
  //   mesh.reconnect();
  // }
  //   Calculate averages
  mesh.updateAveragesAndPlasticEvents();
  mesh.updateCom();
  mesh.updateBoundingBox();
  updateEnergyHistory(true);
  computeParticipationFraction();
  computeM3ParticipationFraction();

  // Updates progress
  m_updateProgress();
  writeToFile();

  // reset some counters
  mesh.resetCounters();

  assert(initialized); // Make sure the simulation has been initialized
}

void Simulation::computeParticipationFraction() {
  if (beforeMinimization.displacements.empty() ||
      afterMinimization.displacements.empty()) {
    participationFraction = 0;
    return;
  }
  const size_t nodeCount = static_cast<size_t>(mesh.rows * mesh.cols);
  if (nodeCount == 0) {
    participationFraction = 0;
    return;
  }

  const size_t expectedSize = 2 * nodeCount;
  if (beforeMinimization.displacements.size() != expectedSize ||
      afterMinimization.displacements.size() != expectedSize) {
    throw std::runtime_error(
        "Snapshot displacements size does not match node count.");
  }

  const double *beforeX = beforeMinimization.displacements.data();
  const double *beforeY = beforeMinimization.displacements.data() + nodeCount;
  const double *afterX = afterMinimization.displacements.data();
  const double *afterY = afterMinimization.displacements.data() + nodeCount;

  double sumU2 = 0.0;
  double sumU4 = 0.0;
  for (size_t i = 0; i < nodeCount; i++) {
    const double dx = afterX[i] - beforeX[i];
    const double dy = afterY[i] - beforeY[i];
    const double u2 = dx * dx + dy * dy; // ||U||^2 per node
    sumU2 += u2;
    sumU4 += u2 * u2;
  }

  if (sumU4 == 0.0) {
    participationFraction = 0;
    return;
  }

  const double N = static_cast<double>(nodeCount);
  participationFraction = (sumU2 * sumU2) / (N * sumU4);
}

void Simulation::computeM3ParticipationFraction() {
  if (beforeMinimization.displacements.empty() ||
      afterMinimization.displacements.empty()) {
    m3ParticipationFraction = 0;
    return;
  }
  const size_t nodeCount = static_cast<size_t>(mesh.rows * mesh.cols);
  if (nodeCount == 0) {
    m3ParticipationFraction = 0;
    return;
  }

  const size_t expectedSize = 2 * nodeCount;
  if (beforeMinimization.displacements.size() != expectedSize ||
      afterMinimization.displacements.size() != expectedSize) {
    throw std::runtime_error(
        "Snapshot displacements size does not match node count.");
  }

  const double *beforeX = beforeMinimization.displacements.data();
  const double *beforeY = beforeMinimization.displacements.data() + nodeCount;
  const double *afterX = afterMinimization.displacements.data();
  const double *afterY = afterMinimization.displacements.data() + nodeCount;

  double sumU2 = 0.0;
  double sumU4 = 0.0;
  size_t changedCount = 0;
  for (const TElement &e : mesh.elements) {
    if (e.m3Nr_fix == e.pastM3Nr_fix) {
      continue;
    }

    Vector2d du = Vector2d::Zero();
    for (const GhostNode &gn : e.ghostNodes) {
      const int nodeIndex = gn.referenceId.i;
      if (nodeIndex >= 0 && static_cast<size_t>(nodeIndex) < nodeCount) {
        const size_t idx = static_cast<size_t>(nodeIndex);
        du[0] += afterX[idx] - beforeX[idx];
        du[1] += afterY[idx] - beforeY[idx];
      }
    }
    du /= 3.0;
    const double u2 = du.squaredNorm();
    sumU2 += u2;
    sumU4 += u2 * u2;
    changedCount += 1;
  }

  if (changedCount == 0 || sumU4 == 0.0) {
    m3ParticipationFraction = 0;
    return;
  }

  const double N = static_cast<double>(changedCount);
  m3ParticipationFraction = (sumU2 * sumU2) / (N * sumU4);
}

void Simulation::updateEnergyHistory(bool endOfStep) {
  SimulationEnergyHistory &h = energyHistory;
  const double totalEnergy = mesh.totalEnergy;
  const double averageEnergy = mesh.averageEnergy;
  const double averageSigma11 = mesh.averageSigma11;
  const double averageSigma12 = mesh.averageSigma12;
  const double averageSigma22 = mesh.averageSigma22;
  const double averageP11 = mesh.averageP11;
  const double averageP12 = mesh.averageP12;
  const double averageP21 = mesh.averageP21;
  const double averageP22 = mesh.averageP22;

  if (mesh.nrMinFunctionCalls == 0 && !endOfStep) {
    // Update initial guess energy values at the start of the minimization
    // Note these are after the first minimization step.
    h.initialGuessTotalEnergy = totalEnergy;
    h.initialGuessAverageEnergy = averageEnergy;
    h.initialGuessAverageSigma11 = averageSigma11;
    h.initialGuessAverageSigma12 = averageSigma12;
    h.initialGuessAverageSigma22 = averageSigma22;
    h.initialGuessAverageP11 = averageP11;
    h.initialGuessAverageP12 = averageP12;
    h.initialGuessAverageP21 = averageP21;
    h.initialGuessAverageP22 = averageP22;
  }

  h.totalEnergyChangeFromInitialGuess = totalEnergy - h.initialGuessTotalEnergy;
  h.averageEnergyChangeFromInitialGuess =
      averageEnergy - h.initialGuessAverageEnergy;
  h.averageSigma12ChangeFromInitialGuess =
      averageSigma12 - h.initialGuessAverageSigma12;

  h.loadStepTotalEnergyChange = totalEnergy - h.prevLoadStepTotalEnergy;
  h.loadStepAverageEnergyChange = averageEnergy - h.prevLoadStepAverageEnergy;
  if (mesh.loadSteps == 1) {
    h.loadStepTotalEnergyChange = 0;
    h.loadStepAverageEnergyChange = 0;
  }

  if (endOfStep) {
    // Prepare for next load step
    h.minIterTotalEnergyChange = 0;
    h.minIterAverageEnergyChange = 0;
    h.prevMinIterTotalEnergy = totalEnergy;
    h.prevMinIterAverageEnergy = averageEnergy;

    h.prevLoadStepTotalEnergy = totalEnergy;
    h.prevLoadStepAverageEnergy = averageEnergy;
  } else {
    // We are inside the minimization loop. Track changes between successive
    // minimization iterations so logDuringMinimization can report per-iteration
    // energy drops.
    if (mesh.nrMinFunctionCalls == 0) {
      // First call in this minimization: initialize the baseline so the first
      // delta is zero.
      h.minIterTotalEnergyChange = 0;
      h.minIterAverageEnergyChange = 0;
      h.prevMinIterTotalEnergy = totalEnergy;
      h.prevMinIterAverageEnergy = averageEnergy;
    } else {
      // Subsequent calls: compute delta relative to previous iteration and
      // advance the baseline.
      h.minIterTotalEnergyChange = totalEnergy - h.prevMinIterTotalEnergy;
      h.minIterAverageEnergyChange = averageEnergy - h.prevMinIterAverageEnergy;
      h.prevMinIterTotalEnergy = totalEnergy;
      h.prevMinIterAverageEnergy = averageEnergy;
    }
  }
}

void Simulation::logMinimizationState() {
  if (!config.logDuringMinimization) {
    return;
  }
  mesh.updateAveragesAndPlasticEvents();
  updateEnergyHistory(false);
  writeToCsv(minCsvFile, *this);
}

void Simulation::m_loadConfig(Config config_) {
  // We save this for serialization
  config = config_;
  setQuiet(isQuiet() || config.showProgress == -1);
  // We fix the random seed to get reproducable results
  setSeed(config.seed);
  // Set the the number of threads
  int maxThreads = omp_get_max_threads();
  int suggestedThreads = std::max(1, static_cast<int>(maxThreads * 0.75));

  if (config.nrThreads == 0) {
    config.nrThreads = suggestedThreads;
  } else if (config.nrThreads > maxThreads) {
    if (!isQuiet()) {
      std::cout << "Too many threads! Wanted " << config.nrThreads
                << ", but only " << maxThreads
                << " are available. Reducing nrThreads to: " << suggestedThreads
                << std::endl;
    }
    config.nrThreads = suggestedThreads;
  }

  omp_set_num_threads(config.nrThreads);
  // This dissables nested loops. We do not want any of these to be
  // happening.
  omp_set_max_active_levels(1);

  // Assign values from Config to Simulation members
  simName = config.name;
  mesh.simName = simName;
  mesh.energyFunction = config.energyFunction;
  mesh.bulkModulus = config.bulkModulus;
  mesh.updateLatticeBasis();
  rows = config.rows;
  cols = config.cols;

  startLoad = config.startLoad;
  loadIncrement = config.loadIncrement;
  maxLoad = config.maxLoad;
  // This flag prevents mesh.reconnect() from being called in the
  // minimization function. If mehs.reconnect() is called somewhere else,
  // reconnection will still occur.
  reconnectionMethod = config.reconnectionMethod;
}

void Simulation::finishSimulation() {
  // gatherDataFiles(); // gather files is run in m_writeDump
  m_writeDump(true);
  csvFile.flush(); // Update the file on disk
  minCsvFile.flush();
  // timer.PrintAllRuntimes();
}

void Simulation::gatherDataFiles() {
  // This creates a pvd file that links all the utv files together.
  createCollection(getDataPath(simName, dataPath),
                   getOutputPath(simName, dataPath));
}

std::string Simulation::saveSimulation(std::string fileName_) {
  timer.Save();
  std::string fileName;

  // Generate filename dynamically if not provided
  if (fileName_.empty()) {
    double range = std::abs(maxLoad - startLoad);
    int precision =
        std::max(1, static_cast<int>(std::ceil(-std::log10(range / 10.0))));
    double roundedLoad = std::round(mesh.load * std::pow(10.0, precision)) /
                         std::pow(10.0, precision);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << roundedLoad;
    fileName = "dump_l" + oss.str() + ".xml.gz"; // Use .zip extension
  } else {
    fileName = fileName_ + ".xml.gz";
  }

  std::string dumpPath =
      std::filesystem::path(getDumpPath(simName, dataPath)) / fileName;

  saveToFile(dumpPath, *this);

  if (!isQuiet()) {
    std::cout << "Dump saved to: " << dumpPath << std::endl;
  }
  return dumpPath;
}

void Simulation::loadSimulation(Simulation &s, const std::string &dumpPath,
                                const std::string &conf, std::string outputPath,
                                std::optional<bool> forceReRun) {

  namespace fs = std::filesystem; // Alias for filesystem

  // Extract XML from .gz or .xml
  loadFromFile(dumpPath, s);
  s.initialized = false;

  // Load config file if provided
  if (!conf.empty()) {
    s.config = parseConfigFile(conf);
    if (s.rows != s.config.rows || s.cols != s.config.cols) {
      throw std::invalid_argument("Mesh size mismatch: Loaded (" +
                                  std::to_string(s.rows) + ", " +
                                  std::to_string(s.cols) + ") vs Config (" +
                                  std::to_string(s.config.rows) + ", " +
                                  std::to_string(s.config.cols) + ")");
    }
  }

  bool quiet = (isQuiet() || s.config.showProgress == -1);
  if (!quiet) {
    std::cout << "Loading simulation from " << dumpPath << std::endl;
  }

  // Update output path
  if (!outputPath.empty()) {
    s.dataPath = outputPath;
  }
  // Test of output path still exsists
  if (!fs::exists(s.dataPath)) {
    std::string oldPath = s.dataPath;
    if (!quiet) {
      std::cout << "Old output path is not found!" << std::endl;
    }

    s.dataPath = findOutputPath();
    if (!s.dataPath.empty()) {
      if (!quiet) {
        std::cout << "Old path:\n\t" << oldPath << "\nNew path:\n\t"
                  << s.dataPath << std::endl;
      }

      // Ask user for confirmation
      char choice;
      if (!quiet) {
        std::cout << "Do you want to continue with the new path? (y/n): "
                  << std::endl;
        std::cin >> choice;
      } else {
        choice = 'y';
      }

      if (choice != 'y' && choice != 'Y') {
        if (!quiet) {
          std::cout << "Aborting due to user rejection of new path."
                    << std::endl;
        }
        throw std::runtime_error("Old output path not found!");
      }
    }
  }

  quiet = (isQuiet() || s.config.showProgress == -1);
  if (!quiet) {
    std::cout << "Loading config..." << std::endl;
  }
  s.m_loadConfig(s.config);

  // Ensure mesh uses the resolved output path (e.g. after -o)
  s.mesh.setSimNameAndDataPath(s.simName, s.dataPath);

  // Use the parameter if provided, otherwise use the one from the config file
  bool effectiveForceReRun =
      forceReRun.has_value() ? *forceReRun : s.config.forceReRun;
  if (s.mesh.load + s.loadIncrement / 2 >= s.maxLoad && !effectiveForceReRun) {
    throw SimulationAlreadyComplete("Simulation already complete");
  }
  if (!quiet) {
    std::cout << "Saving config..." << std::endl;
  }
  saveConfigFile(s.config, s.dataPath);
  s.addDefaultCsvColumns();
  {
    const std::filesystem::path csvPath =
        std::filesystem::path(getOutputPath(s.simName, s.dataPath)) /
        (MACRODATANAME + std::string(".csv"));
    s.recoverCsvColumnsFromFile(csvPath.string());
  }
  s.csvFile = initCsvFile(s.simName, s.dataPath, s);

  if (!quiet) {
    std::cout << "Initializing..." << std::endl;
  }
  s.initialize();
  s.mesh.updateMesh();
  s.mesh.updateAveragesAndPlasticEvents();
  s.mesh.updateAngles();
  if (!quiet) {
    std::cout << "Simulation loaded!" << std::endl;
  }
}

void Simulation::addReversibilityCsvColumns() {
  const auto addIfMissing = [this](const std::string &name, auto accessor) {
    if (!hasCsvColumn(name)) {
      addCsvColumn(name, accessor);
    }
  };

  addIfMissing("is_reversible", [](const Simulation &s) {
    return s.reversibilityState.isReversible;
  });

  addIfMissing("rev_u_diff", [](const Simulation &s) {
    return s.reversibilityState.distance;
  });

  addIfMissing("rev_energy_diff", [](const Simulation &s) {
    return s.reversibilityState.energyDifference;
  });

  addIfMissing("rev_sigma_12_diff", [](const Simulation &s) {
    return s.reversibilityState.sigma12Difference;
  });

  addIfMissing("rev_sigma_trace_diff", [](const Simulation &s) {
    return s.reversibilityState.sigmaTraceDifference;
  });

  addIfMissing("rev_sigma11_diff", [](const Simulation &s) {
    return s.reversibilityState.sigma11Difference;
  });

  addIfMissing("rev_sigma22_diff", [](const Simulation &s) {
    return s.reversibilityState.sigma22Difference;
  });

  addIfMissing("rev_p11_diff", [](const Simulation &s) {
    return s.reversibilityState.p11Difference;
  });

  addIfMissing("rev_p12_diff", [](const Simulation &s) {
    return s.reversibilityState.p12Difference;
  });

  addIfMissing("rev_p21_diff", [](const Simulation &s) {
    return s.reversibilityState.p21Difference;
  });

  addIfMissing("rev_p22_diff", [](const Simulation &s) {
    return s.reversibilityState.p22Difference;
  });
}

bool Simulation::checkReversibility(const Matrix2d &stepTransform, double eps) {
  Mesh::Snapshot state0 = mesh.snapshotState();

  // Save initial energy and sigma values for later comparison
  const double initialEnergy = mesh.totalEnergy;
  const double initialSigma12 = mesh.averageSigma12;
  const double initialSigma11 = mesh.averageSigma11;
  const double initialSigma22 = mesh.averageSigma22;
  const double initialSigmaTrace = mesh.averageSigmaTrace;
  const double initialP11 = mesh.averageP11;
  const double initialP12 = mesh.averageP12;
  const double initialP21 = mesh.averageP21;
  const double initialP22 = mesh.averageP22;

  // Forward step (0 -> 1 -> 2)
  applyAffineStep(stepTransform); // (0 -> 1)
  Mesh::Snapshot state1 = mesh.snapshotState();
  mesh.updateMesh();
  const double energyAffine = mesh.totalEnergy;
  minimize(); // (1 -> 2)

  mesh.updateAveragesAndPlasticEvents();
  const bool hasPlasticEvents = mesh.nr_elements_with_m3_fix_changeInStep > 0;
  const bool hasEnergyDrop = mesh.totalEnergy < energyAffine;

  if (!(hasPlasticEvents && hasEnergyDrop)) {
    reversibilityState.isReversible = 1;
    reversibilityState.distance = 0;
    reversibilityState.energyDifference = 0;
    reversibilityState.sigma12Difference = 0;
    reversibilityState.sigmaTraceDifference = 0;
    reversibilityState.sigma11Difference = 0;
    reversibilityState.sigma22Difference = 0;
    reversibilityState.p11Difference = 0;
    reversibilityState.p12Difference = 0;
    reversibilityState.p21Difference = 0;
    reversibilityState.p22Difference = 0;
    return true;
  }

  Mesh::Snapshot state2 = mesh.snapshotState();
  const auto elementsState2 = mesh.elements;
  const int nrElementsState2 = mesh.nrElements;
  const auto currentEdgeSetState2 = mesh.currentEdgeSet;
  const auto previousEdgeSetState2 = mesh.previousStepEdgeSet;
  const std::size_t edgeFlipsSinceLastStepState2 = mesh.edgeFlipsSinceLastStep;
  const auto flipRecordsState2 = mesh.flipRecords;
  const int flipRecordCountState2 = mesh.flipRecordCount;
  const auto flippedThisReconnectState2 = mesh.flippedThisReconnect;
  const bool meshReconnectedState2 = mesh.meshReconnected;
  const int nrM3ChangeState2 = mesh.nr_elements_with_m3_fix_change;
  const int nrM3ChangeInStepState2 = mesh.nr_elements_with_m3_fix_changeInStep;
  SimulationEnergyHistory historyState2 = energyHistory;
  SimReport FIRERepState2 = FIRERep;
  SimReport LBFGSRepState2 = LBFGSRep;
  SimReport CGRepState2 = CGRep;
  int nrMinItterationsState2 = mesh.nrMinItterations;
  int nrMinFunctionCallsState2 = mesh.nrMinFunctionCalls;

  // Backward step (2 -> 3 -> 4)
  const double oldIncrement = loadIncrement;
  loadIncrement = -oldIncrement;
  const Matrix2d backTransform = stepTransform.inverse();
  applyAffineStep(backTransform);
  Mesh::Snapshot state3 = mesh.snapshotState();
  minimize();

  Mesh::Snapshot state4 = mesh.snapshotState();
  const double d = mesh.rmsDistanceToSnapshot(state0, true);

  // Save reversibility results
  reversibilityState.distance = d;
  reversibilityState.energyDifference =
      std::abs(mesh.totalEnergy - initialEnergy);
  reversibilityState.sigma12Difference =
      std::abs(mesh.averageSigma12 - initialSigma12);
  reversibilityState.sigmaTraceDifference =
      std::abs(mesh.averageSigmaTrace - initialSigmaTrace);
  reversibilityState.sigma11Difference =
      std::abs(mesh.averageSigma11 - initialSigma11);
  reversibilityState.sigma22Difference =
      std::abs(mesh.averageSigma22 - initialSigma22);
  reversibilityState.p11Difference = std::abs(mesh.averageP11 - initialP11);
  reversibilityState.p12Difference = std::abs(mesh.averageP12 - initialP12);
  reversibilityState.p21Difference = std::abs(mesh.averageP21 - initialP21);
  reversibilityState.p22Difference = std::abs(mesh.averageP22 - initialP22);

  reversibilityState.isReversible = d < eps ? 1 : 0;
  const bool reversible = d < eps;

  // Save drop snapshots (every 300 reversible drops, every 10 irreversible).
  static int reversibleDropCount = 0;
  static int irreversibleDropCount = 0;
  auto saveDrop = [&](const std::string &folder) {
    const std::string dropSubFolder =
        std::string("reversibilityData/") + folder;
    const std::string dropPath =
        getDataPath(simName, dataPath) + "/" + dropSubFolder;
    auto writeSnap = [&](const Mesh::Snapshot &snap, const std::string &name) {
      mesh.restoreState(snap);
      mesh.ensureFull();
      writeMeshToVtu(mesh, simName, dataPath, name, false, VtuFieldLevel::All,
                     "", dropSubFolder);
    };

    Mesh::Snapshot current = mesh.snapshotState();
    writeSnap(state0, "state0_min_gamma");
    writeSnap(state1, "state1_affine_gamma_plus");
    writeSnap(state2, "state2_relaxed_gamma_plus");
    writeSnap(state3, "state3_affine_gamma_minus");
    writeSnap(state4, "state4_relaxed_gamma");
    createCollection(dropPath, dropPath);
    mesh.restoreState(current);
  };

  const double step = std::abs(oldIncrement);
  assert(step > 0);
  int precision = std::max(0, static_cast<int>(std::ceil(-std::log10(step))));
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << mesh.load;
  if (reversible) {
    reversibleDropCount++;
    if (reversibleDropCount % 300 == 0) {
      saveDrop("rev_drop_l_" + oss.str());
    }
  } else {
    irreversibleDropCount++;
    if (irreversibleDropCount % 10 == 0) {
      saveDrop("irrev_drop_l_" + oss.str());
    }
  }

  // Restore to state 2 (forward relaxed state)
  mesh.elements = elementsState2;
  mesh.nrElements = nrElementsState2;
  mesh.rebuildConnectivity();
  mesh.currentEdgeSet = currentEdgeSetState2;
  mesh.previousStepEdgeSet = previousEdgeSetState2;
  mesh.edgeFlipsSinceLastStep = edgeFlipsSinceLastStepState2;
  mesh.flipRecords = flipRecordsState2;
  mesh.flipRecordCount = flipRecordCountState2;
  mesh.flippedThisReconnect = flippedThisReconnectState2;
  mesh.meshReconnected = meshReconnectedState2;
  mesh.nr_elements_with_m3_fix_change = nrM3ChangeState2;
  mesh.nr_elements_with_m3_fix_changeInStep = nrM3ChangeInStepState2;
  mesh.restoreState(state2);
  energyHistory = historyState2;
  FIRERep = FIRERepState2;
  LBFGSRep = LBFGSRepState2;
  CGRep = CGRepState2;
  mesh.nrMinItterations = nrMinItterationsState2;
  mesh.nrMinFunctionCalls = nrMinFunctionCallsState2;
  loadIncrement = oldIncrement;

  return reversible;
}

// This can be used in the LBFGS optimization
void iterationLogger(const alglib::real_1d_array &x, double energy,
                     void *dataLinkPtr) {
  (void)x; // Explicitly casting x to void to silence unused parameter warning

  DataLink *dataLink = reinterpret_cast<DataLink *>(dataLinkPtr);

  // dataPath looks like: "/Volumes/data/MTS2D_output/"

  Mesh *mesh = dataLink->mesh;

  // auto state = dataLink->minState;

  // Increment the iteration count
  // TODO Does not work for CG or FIRE, only for LBFGS
  int it = dataLink->LBFGS_state->c_ptr()->repiterationscount;
  int nrFc = mesh->nrMinFunctionCalls;

  // Check if iteration count is a multiple of 5000
  if (nrFc % 5000 == 0 && nrFc > 0) {
    if (!isQuiet()) {
      std::cout << "Warning (step " << mesh->loadSteps << "): " << it
                << " iterations and " << nrFc << " function calls."
                << std::endl;
    }
  }

  // Save when energy shifts by >10% vs last saved, otherwise every 1000
  // function calls.
  int saveEvery = 1000;
  static int lastSavedFc = -1;
  static double lastSavedEnergy = std::numeric_limits<double>::quiet_NaN();

  if (dataLink->s->config.logDuringMinimization) {
    dataLink->s->timer.Start("write");
    // Write to the CSV file
    dataLink->s->logMinimizationState();
    bool energyJump = false;
    if (std::isnan(lastSavedEnergy)) {
      energyJump = true;
    } else {
      double denom = std::max(std::abs(lastSavedEnergy), 1e-12);
      energyJump = std::abs(energy - lastSavedEnergy) / denom > 0.10;
    }

    bool stepSave = (lastSavedFc < 0) || (abs(nrFc - lastSavedFc) >= saveEvery);

    if (energyJump || stepSave) {
      // Write mesh to file
      mesh->writeToVtu("", true, VtuFieldLevel::Minimal);
      lastSavedFc = nrFc;
      lastSavedEnergy = energy;
    }
    dataLink->s->timer.Stop("write");
  }
}

Matrix2d getShear(double load, double theta) {
  // perturb is currently unused. If it will be used, it should be
  // implemeted propperly.
  double perturb = 0;

  Matrix2d trans;
  trans(0, 0) = (1. - load * cos(theta + perturb) * sin(theta + perturb));
  trans(1, 1) = (1. + load * cos(theta - perturb) * sin(theta - perturb));
  trans(0, 1) = load * pow(cos(theta), 2.);
  trans(1, 0) = -load * pow(sin(theta - perturb), 2.);

  return trans;
}

DataLink::DataLink(Simulation *simulation) {
  s = simulation;
  mesh = &s->mesh;
  stopSignal = nullptr; // This is set right before minimization
  minState = &s->minState;
  LBFGS_state = &s->LBFGS_state;
  CG_state = &s->CG_state;
  maxForceAllowed = &s->config.epsR;
}
