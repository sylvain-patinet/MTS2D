#include "mesh.h"
#include "Data/data_export.h"
#include "Mesh/node.h"
#include "Mesh/tElement.h"
#include "Simulation/randomUtils.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using Eigen::Vector2i;
// CGAL is a heavy library and slows down clangd and other tools a lot.
// We use a seperate build without CGAL.
#if !defined(IDE_LIGHTWEIGHT)
#include <CGAL/Delaunay_triangulation_2.h>
// I had some issues with inexact constructions giving different triangulations
// where they should be the same.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/centroid.h>

using K = CGAL::Exact_predicates_exact_constructions_kernel; // exact predicates
                                                             // & constructions
using Point = K::Point_2;
using Delaunay = CGAL::Delaunay_triangulation_2<K>;
// Add this just above the CGAL aliases
struct VInfo {
  int refNodeIndex; // index of the reference node in the MTS mesh
                    // (0..nrNodes-1)
};
using Vb = CGAL::Triangulation_vertex_base_with_info_2<VInfo, K>;
using Fb = CGAL::Triangulation_face_base_2<K>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using DelaunayInfo = CGAL::Delaunay_triangulation_2<K, Tds>;
#endif

Mesh::Mesh() {}

// Constructor that initializes the surface with size n x m
Mesh::Mesh(int rows, int cols, double a, double QDSD, bool usingPBC,
           std::string diagonal, std::string energyFunction, double bulkModulus)
    : nodes(rows, cols), a(a), rows(rows), cols(cols), loadSteps(0),
      currentDeformation(Eigen::Matrix2d::Identity()), QDSD(QDSD),
      energyFunction(energyFunction), bulkModulus(bulkModulus),
      usingPBC(usingPBC), diagonal(diagonal) {
  // Calculate nrElements based on whether usingPBC is true
  if (usingPBC) {
    nrElements = 2 * rows * cols;
  } else {
    nrElements = 2 * (rows - 1) * (cols - 1);
  }
  nrNodes = rows * cols;

  // Now initialize elements with the calculated size
  elements.resize(nrElements);
  resetFlipTracking();

  updateLatticeBasis();
  m_createNodes();
  m_updateFixedAndFreeNodeIds();

  // we create some elements here, but note that these will need to be replaced
  // if changes are made to the fixed and free status of the nodes.
  createElements();
  updateAveragesAndPlasticEvents();
  resetCounters();
}

Mesh::Mesh(int rows, int cols, bool usingPBC, std::string diagonal)
    : Mesh(rows, cols, 1, 0, usingPBC, diagonal) {}
Mesh::Mesh(int rows, int cols, bool usingPBC)
    : Mesh(rows, cols, usingPBC, "major") {}

void Mesh::updateLatticeBasis() {
  if (energyFunction == "conti_triangular" ||
      energyFunction == "contiTriangular") {
    const double h = std::sqrt(3.0) * 0.5 * a;
    latticeBasis << a, 0.5 * a, 0.0, h;
  } else {
    latticeBasis = Matrix2d::Identity() * a;
  }
  // Keep a fixed reference area for each element to preserve mass even if the
  // reference triangle collapses after reconnecting.
  init_element_area = 0.5 * std::abs(latticeBasis.determinant());
  markDirty();
}

bool Mesh::isFixedNode(const NodeId &nodeId) const {
  return (*this)[nodeId]->fixedNode;
}

void Mesh::addLoad(double loadChange) {
  load += loadChange;
  loadSteps++;
}

void Mesh::applyTransformation(const Matrix2d &transformation) {
  // We get all the nodes in the mesh.
  for (long i = 0; i < nodes.size(); i++) {
    transformInPlace(transformation, nodes(i));
  }
  // We also assume we want to transform the current deformation
  applyTransformationToSystemDeformation(transformation);
  markDirty();
}

void Mesh::applyTransformationToFixedNodes(const Matrix2d &transformation) {
  // We get the id of each node in the border
  for (NodeId &nodeId : fixedNodeIds) {
    transformInPlace(transformation, *(*this)[nodeId]);
  }
  markDirty();
}

void Mesh::applyTransformationToSystemDeformation(
    const Matrix2d &transformation) {
  currentDeformation = transformation * currentDeformation;
  markDirty();
}

void Mesh::applyTranslation(const Vector2d &displacement) {
  for (long i = 0; i < nodes.size(); i++) {
    translateInPlace(nodes(i), displacement);
  }
  markDirty();
}

void Mesh::setRefConfiguration() {
  for (long i = 0; i < nodes.size(); i++) {
    nodes(i).setRefPos(nodes(i).pos());
  }
  markDirty();
}

static inline int wrap_index(int i, int n) {
  int r = i % n;
  return r < 0 ? r + n : r;
}

Node *Mesh::getNode(const Vector2i &rowCol) {
  int x = rowCol.x(); // col
  int y = rowCol.y(); // row
  // Always assume PBC
  x = wrap_index(x, cols);
  y = wrap_index(y, rows);

  // nodes(row, col) -> (y, x)
  return &nodes(y, x);
}

GhostNode Mesh::getGhostNode(const Vector2i &rowCol) {
  int x = rowCol.x(); // col
  int y = rowCol.y(); // row
  // Always assume PBC
  int ref_x = wrap_index(x, cols);
  int ref_y = wrap_index(y, rows);

  // nodes(row, col) -> (y, x)
  Node *n = &nodes(ref_y, ref_x);
  return m_gn(n, y, x);
}

Node *Mesh::getNeighbourNode(const Node &node, const Vector2i &direction) {
  int x = node.id.col() + direction.x(); // col
  int y = node.id.row() + direction.y(); // row
  if (!usingPBC) {
    // Check bounds
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
      throw std::out_of_range(
          "Attempted to access neighbour node out of mesh bounds.");
    }
  }
  return getNode(Vector2i{x, y});
}

// Function to fix the elements of the border vector
void Mesh::fixBorderNodes() {
  fixNodesInRow(0);
  fixNodesInColumn(0);
  fixNodesInRow(rows - 1);
  fixNodesInColumn(cols - 1);
  m_updateFixedAndFreeNodeIds();
}

void Mesh::fixBottomLeftCorner() {
  // This is useful in PBC to avoid translation (and maybe rotation?)

  // Fix the bottom left corner node
  nodes(0, 0).fixedNode = true;
  // Update the fixed and free node IDs
  m_updateFixedAndFreeNodeIds();
}

void Mesh::fixNodesInRow(int row) {
  // Allow for negative indexing
  if (row < 0) {
    row = cols + row;
  }
  for (int col = 0; col < cols; ++col) {
    nodes(row, col).fixedNode = true;
  }
  m_updateFixedAndFreeNodeIds();
}

void Mesh::fixNodesInColumn(int col) {
  // Allow for negative indexing
  if (col < 0) {
    col = cols + col;
  }
  for (int row = 0; row < rows; ++row) {
    nodes(row, col).fixedNode = true;
  }
  m_updateFixedAndFreeNodeIds();
}

void Mesh::m_updateFixedAndFreeNodeIds() {
  fixedNodeIds.clear();
  freeNodeIds.clear();
  for (long i = 0; i < nodes.size(); i++) {
    NodeId nodeId(i, cols);
    if (isFixedNode(nodeId)) {
      fixedNodeIds.push_back(nodeId);
    } else {
      freeNodeIds.push_back(nodeId);
    }
  }
}

void Mesh::m_createNodes() {
  int n = rows;
  int m = cols;

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < m; ++col) {
      Vector2d pos = latticeBasis * Vector2d(col, row);
      Node node(pos.x(), pos.y());
      node.id = NodeId(row, col, m);
      nodes(row, col) = node;
    }
  }
}

// Gets the four nodes and their ghost versions for a given row and column in
// the reference state. NOT in the current state.
std::vector<Node *> Mesh::getSquareNodes(int row, int col) {
  // We find the 4 nodes in the current square
  Node *n1 = (*this)[m_makeNId(row, col)];
  Node *n2 = getNeighbourNode(*n1, RIGHT_N);
  Node *n3 = getNeighbourNode(*n1, UP_N);
  // n4 is now up AND right of n1
  Node *n4 = getNeighbourNode(*n2, UP_N);

  // The nodes should be in this square configuration in the reference state
  // n3  n4
  // n1  n2

  // This function will find nodes across the system if necessary due to
  // periodic boundaries
  return {n1, n2, n3, n4};
}

// Gets the four nodes and their ghost versions for a given row and column
std::vector<GhostNode> Mesh::getSquareGhostNodes(int row, int col) {
  auto nodes = getSquareNodes(row, col);
  return m_makeGhostNodes(nodes, row, col);
}

// Calculates element indices for a given row and column
std::pair<int, int> Mesh::getElementIndices(int row, int col) {
  int e1i = 2 * (row * ePairCols() + col); // Triangle 1 index
  int e2i = e1i + 1;                       // Triangle 2 index
  return {e1i, e2i};
}

/**
 * Creates or updates two triangular elements based on the specified diagonal
 * direction
 * @param ghosts array of four ghost nodes {g1, g2, g3, g4}
 * @param e1i Index of first element
 * @param e2i Index of second element
 * @param useLeftDiagonal Whether to use left diagonal splitting
 * @param preserveNoise Whether to preserve existing noise values (true for
 * updates)
 */
void Mesh::createElementPair(const std::vector<GhostNode> &ghosts, int e1i,
                             int e2i, bool useMajorDiagonal,
                             bool preserveNoise) {

  // The nodes should be in this square configuration in the current state
  // g3  g4
  // g1  g2
  assert(ghosts.size() == 4);

  GhostNode g1 = ghosts[0];
  GhostNode g2 = ghosts[1];
  GhostNode g3 = ghosts[2];
  GhostNode g4 = ghosts[3];

  double noise1, noise2;

  if (preserveNoise) {
    noise1 = elements[e1i].noise;
    noise2 = elements[e2i].noise;
  } else {
    noise1 = sampleNormal(1, QDSD);
    noise2 = sampleNormal(1, QDSD);
  }

  // When choosing what order to give the nodes to the element, we carefully
  // choose the first node to be the corner node, such that, in the reference
  // frame, all elements have an angle of 90 degrees.

  if (useMajorDiagonal) {
    // Split using major-diagonal from top-left to bottom-right (↘)
    elements[e1i] =
        TElement((*this), g1, g2, g3, e1i, noise1, energyFunction, bulkModulus);
    elements[e2i] =
        TElement((*this), g4, g2, g3, e2i, noise2, energyFunction, bulkModulus);
  } else {
    // Split using minor-diagonal from top-right to bottom-left (↙)
    elements[e1i] =
        TElement((*this), g2, g1, g4, e1i, noise1, energyFunction, bulkModulus);
    elements[e2i] =
        TElement((*this), g3, g1, g4, e2i, noise2, energyFunction, bulkModulus);
  }
}

void Mesh::createElementPair(const std::vector<const GhostNode *> &ghostsPtr,
                             int e1i, int e2i, bool useMajorDiagonal,
                             bool preserveNoise) {

  const std::vector<GhostNode> ghosts = {*ghostsPtr[0], *ghostsPtr[1],
                                         *ghostsPtr[2], *ghostsPtr[3]};
  createElementPair(ghosts, e1i, e2i, useMajorDiagonal, preserveNoise);
}

void Mesh::createElementPair(const std::array<const GhostNode *, 4> &ghostsPtr,
                             int e1i, int e2i, bool useMajorDiagonal,
                             bool preserveNoise) {
  const std::array<GhostNode, 4> ghosts = {*ghostsPtr[0], *ghostsPtr[1],
                                           *ghostsPtr[2], *ghostsPtr[3]};
  // Inline the array version to avoid dynamic allocations.
  const GhostNode &g1 = ghosts[0];
  const GhostNode &g2 = ghosts[1];
  const GhostNode &g3 = ghosts[2];
  const GhostNode &g4 = ghosts[3];

  double noise1, noise2;
  if (preserveNoise) {
    noise1 = elements[e1i].noise;
    noise2 = elements[e2i].noise;
  } else {
    noise1 = sampleNormal(1, QDSD);
    noise2 = sampleNormal(1, QDSD);
  }

  if (useMajorDiagonal) {
    elements[e1i] =
        TElement((*this), g1, g2, g3, e1i, noise1, energyFunction, bulkModulus);
    elements[e2i] =
        TElement((*this), g4, g2, g3, e2i, noise2, energyFunction, bulkModulus);
  } else {
    elements[e1i] =
        TElement((*this), g2, g1, g4, e1i, noise1, energyFunction, bulkModulus);
    elements[e2i] =
        TElement((*this), g3, g1, g4, e2i, noise2, energyFunction, bulkModulus);
  }
}

void Mesh::createElements() {
  // Note that neighbours must be filled before using this function.

  // We construct the elements by finding the four nodes that create two
  // opposing cells, but stopping before we get to the last row and columns if
  // we don't use periodic boundary conditions.
  for (int row = 0; row < ePairRows(); ++row) {
    for (int col = 0; col < ePairCols(); ++col) {
      auto ghosts = getSquareGhostNodes(row, col);
      auto [e1i, e2i] = getElementIndices(row, col);

      // Determine diagonal direction based on alternating pattern

      bool useMajorDiagonal;
      if (diagonal == "major") {
        useMajorDiagonal = true;
      } else if (diagonal == "minor") {
        useMajorDiagonal = false;
      } else if (diagonal == "alternate") {
        useMajorDiagonal = (row + col) % 2;
      } else {
        throw std::invalid_argument("Unkown meshing: " + diagonal);
      }

      createElementPair(ghosts, e1i, e2i, useMajorDiagonal, false);
    }
  }

  rebuildConnectedGhostNodes();
  initializeEdgeSets();
}

// This is just a function to avoid having to write cols
NodeId Mesh::m_makeNId(int row, int col) { return NodeId(row, col, cols); }

void Mesh::resetCounters() {
  nrMinItterations = 0;
  nrMinFunctionCalls = 0;
  if (currentEdgeSet.empty()) {
    throw std::runtime_error("resetCounters: edge sets not initialized");
  }
  previousStepEdgeSet = currentEdgeSet;
  resetPastPlasticCount();
  resetFlipTracking();
}

void Mesh::resetPastPlasticCount(bool endOfStep) {
  nr_elements_with_m3_fix_changeInStep = 0;
  for (size_t i = 0; i < elements.size(); i++) {
    elements[i].pastStepM3Nr_fix = elements[i].m3Nr_fix;
  }

  if (endOfStep) {
    nr_elements_with_m3_fix_change = 0;
    for (size_t i = 0; i < elements.size(); i++) {
      elements[i].pastM3Nr_fix = elements[i].m3Nr_fix;
    }
  }
}

void Mesh::setSimNameAndDataPath(std::string name, std::string path) {
  simName = name;
  dataPath = path;
}

// Helper function to make ghost node

// This function automatically sets the periodic shift based on row and col
GhostNode Mesh::m_gn(const Node *n, int row, int col) {
  return GhostNode(n, row, col, cols, latticeBasis, currentDeformation);
}
// This function automatically sets the periodic shift based on targetPos
GhostNode Mesh::m_gn(const Node *n, const Vector2d targetPos) {
  return GhostNode(n, targetPos, rows, cols, latticeBasis, currentDeformation);
}
GhostNode Mesh::m_gn(const Node *n) {
  return GhostNode(n, n->id.row(), n->id.col(), cols, latticeBasis,
                   currentDeformation);
}

// The idea here is that we have taken our grid, and made rows and columns of
// squares of 4 and 4 nodes. This function converts from reference nodes to
// ghost nodes and makes sure that the ghost nodes are appropreately shifted
// when that is requried.
// We never want our square to span the entire system which would happen at the
// boundary if we use the "real" position of the nodes
std::vector<GhostNode>
Mesh::m_makeGhostNodes(const std::vector<Node *> refNodes, int row, int col) {
  assert(refNodes.size() == 4);

  const Node *n1 = refNodes[0];
  const Node *n2 = refNodes[1];
  const Node *n3 = refNodes[2];
  const Node *n4 = refNodes[3];

  GhostNode gn1 = m_gn(n1);
  GhostNode gn2 = m_gn(n2);
  GhostNode gn3 = m_gn(n3);
  GhostNode gn4 = m_gn(n4);

  if (usingPBC) {

    if (row == rows - 1 && col == cols - 1) {
      // If we are in the corner, we need to move n2, n3 and n4
      gn2 = m_gn(n2, n2->id.row(), cols);
      gn3 = m_gn(n3, rows, n3->id.col());
      gn4 = m_gn(n4, rows, cols);
    } else if (col == cols - 1) {
      // If we are in the last column, we need to move n2 and n4
      gn2 = m_gn(n2, n2->id.row(), cols);
      gn4 = m_gn(n4, n4->id.row(), cols);
    } else if (row == rows - 1) {
      // If we are in the last row, we need to move n3 and n4
      gn3 = m_gn(n3, rows, n3->id.col());
      gn4 = m_gn(n4, rows, n4->id.col());
    }
  }

  return {gn1, gn2, gn3, gn4};
}

void Mesh::printConnectivity(bool realId) {
  std::string sep;
  std::string end;
  if (nrNodes <= 9) {
    sep = "";
    end = " ";
  } else {
    sep = ",";
    end = "\n";
  }
  for (int i = 0; i < nrElements; i++) {
    TElement &e = elements[i];
    for (size_t j = 0; j < e.ghostNodes.size(); j++) {
      if (realId) {
        std::cout << e.ghostNodes[j].referenceId.i << sep;
      } else {
        std::cout << e.ghostNodes[j].id << sep;
      }
    }
    std::cout << end;
  }
  std::cout << '\n';
}

// Updates the forces on the nodes in the surface and returns the total
// energy from all the elements in the surface.
void Mesh::updateMesh() { ensureForces(); }

void Mesh::updateElements() { ensureForces(); }

void Mesh::markDirty() { updateState = UpdateState::Dirty; }

void Mesh::ensureForces() {
  if (updateState != UpdateState::Dirty) {
    return;
  }
  updateElementsForces();
  updateState = UpdateState::Forces;
}

void Mesh::ensureGeometry() {
  if (updateState == UpdateState::Geometry ||
      updateState == UpdateState::Full) {
    return;
  }
  ensureForces();
  updateElementsGeometry();
  updateState = UpdateState::Geometry;
}

void Mesh::ensureFull() {
  if (updateState == UpdateState::Full) {
    return;
  }
  ensureGeometry();
  updateElementsFull();
  updateState = UpdateState::Full;
}

/*
This function has given me a lot of headache. When multithreading, the program
will sometimes hang, with all threads waiting for each other. I have checked
that all threads do actually reach the barrier, but the function never returns.
Using a manual barrier, i can add a timeout. This seems to work.
*/
//Change Sylvain
void Mesh::updateElementsForces() {
    omp_set_dynamic(0);

    const int nNodes = nodes.size();
    const int nThreads = omp_get_max_threads();
    const size_t scratchSize = static_cast<size_t>(nThreads) * nNodes;

    if (forceScratch.size() != scratchSize || forceScratchThreads != nThreads) {
        forceScratch.resize(scratchSize);
        forceScratchThreads = nThreads;
    }

    double energy_sum = 0.0;

#pragma omp parallel reduction(+ : energy_sum)
    {
        const int tid = omp_get_thread_num();
        Vector2d *local = forceScratch.data() + static_cast<size_t>(tid) * nNodes;

        // Parallel first-touch / zeroing of this thread's stripe
        for (int i = 0; i < nNodes; ++i) {
            local[i].setZero();
        }

#pragma omp for schedule(static) nowait
        for (int i = 0; i < nrElements; ++i) {
            TElement &e = elements[i];
            e.updateForces(*this);
            energy_sum += e.energy;

            const GhostNode &g0 = e.ghostNodes[0];
            const GhostNode &g1 = e.ghostNodes[1];
            const GhostNode &g2 = e.ghostNodes[2];

            local[g0.referenceId.i] += g0.f;
            local[g1.referenceId.i] += g1.f;
            local[g2.referenceId.i] += g2.f;
        }
    }

    totalEnergy = energy_sum;

    double maxForce = 0.0;
#pragma omp parallel for reduction(max : maxForce)
    for (int i = 0; i < nNodes; ++i) {
        Vector2d sum = Vector2d::Zero();
        for (int t = 0; t < nThreads; ++t) {
            sum += forceScratch[static_cast<size_t>(t) * nNodes + i];
        }
        nodes(i).f = sum;
        if (!nodes(i).fixedNode) {
            maxForce = std::max({maxForce, std::abs(sum[0]), std::abs(sum[1])});
        }
    }

    this->maxForce = maxForce;
}

// void Mesh::updateElementsForces() {
//   omp_set_dynamic(0); // once per process is fine too
//   const int nNodes = nodes.size();
//   const int nThreads = omp_get_max_threads();
//   const size_t scratchSize = static_cast<size_t>(nThreads) * nNodes;
//
//   if (forceScratch.size() != scratchSize || forceScratchThreads != nThreads) {
//     forceScratch.assign(scratchSize, Vector2d::Zero());
//     forceScratchThreads = nThreads;
//   } else {
//     std::fill(forceScratch.begin(), forceScratch.end(), Vector2d::Zero());
//   }
//
//   double energy_sum = 0.0;
//
// #pragma omp parallel
//   {
//     const int tid = omp_get_thread_num();
//     Vector2d *local = forceScratch.data() + static_cast<size_t>(tid) * nNodes;
//
// #pragma omp for schedule(static, 1024) reduction(+ : energy_sum) nowait
//     for (int i = 0; i < nrElements; ++i) {
//       TElement &e = elements[i];
//       e.updateForces(*this); // must not wait on other elements
//       energy_sum += e.energy;
//
//       const GhostNode &g0 = e.ghostNodes[0];
//       const GhostNode &g1 = e.ghostNodes[1];
//       const GhostNode &g2 = e.ghostNodes[2];
//       local[g0.referenceId.i] += g0.f;
//       local[g1.referenceId.i] += g1.f;
//       local[g2.referenceId.i] += g2.f;
//     }
//   }
//   totalEnergy = energy_sum;
//
//   // Update forces on nodes by summing contributions from all elements in all
//   // threads and track max force over free nodes.
//   double maxForce = 0.0;
// #pragma omp parallel for reduction(max : maxForce)
//   for (int i = 0; i < nNodes; ++i) {
//     Vector2d sum = Vector2d::Zero();
//     for (int t = 0; t < nThreads; ++t) {
//       sum += forceScratch[static_cast<size_t>(t) * nNodes + i];
//     }
//     nodes(i).f = sum;
//     if (!nodes(i).fixedNode) {
//       maxForce = std::max({maxForce, std::abs(sum[0]), std::abs(sum[1])});
//     }
//   }
//   this->maxForce = maxForce;
// }
//
// void Mesh::updateElementsForces() {
//     omp_set_dynamic(0);
//
//     const int nNodes = nodes.size();
//     const int nThreads = omp_get_max_threads();
//     const size_t scratchSize = static_cast<size_t>(nThreads) * nNodes;
//
//     if (forceScratch.size() != scratchSize || forceScratchThreads != nThreads) {
//         forceScratch.resize(scratchSize);
//         forceScratchThreads = nThreads;
//     }
//
//     double energy_sum = 0.0;
//
// #pragma omp parallel reduction(+ : energy_sum)
//     {
//         const int tid = omp_get_thread_num();
//         Vector2d *local = forceScratch.data() + static_cast<size_t>(tid) * nNodes;
//
//         // Parallel first-touch / zeroing of this thread's stripe
//         for (int i = 0; i < nNodes; ++i) {
//             local[i].setZero();
//         }
//
// #pragma omp for schedule(static) nowait
//         for (int i = 0; i < nrElements; ++i) {
//             TElement &e = elements[i];
//             e.updateForces(*this);
//             energy_sum += e.energy;
//
//             const GhostNode &g0 = e.ghostNodes[0];
//             const GhostNode &g1 = e.ghostNodes[1];
//             const GhostNode &g2 = e.ghostNodes[2];
//
//             local[g0.referenceId.i] += g0.f;
//             local[g1.referenceId.i] += g1.f;
//             local[g2.referenceId.i] += g2.f;
//         }
//     }
//
//     totalEnergy = energy_sum;
//
//     double maxForce = 0.0;
// #pragma omp parallel for reduction(max : maxForce)
//     for (int i = 0; i < nNodes; ++i) {
//         Vector2d sum = Vector2d::Zero();
//         for (int t = 0; t < nThreads; ++t) {
//             sum += forceScratch[static_cast<size_t>(t) * nNodes + i];
//         }
//         nodes(i).f = sum;
//         if (!nodes(i).fixedNode) {
//             maxForce = std::max({maxForce, std::abs(sum[0]), std::abs(sum[1])});
//         }
//     }
//
//     this->maxForce = maxForce;
// }
//Changes Sylvain



void Mesh::updateElementsGeometry() {
#pragma omp parallel for schedule(static, 1024)
  for (int i = 0; i < nrElements; ++i) {
    elements[i].updateGeometry();
  }
}

void Mesh::updateElementsFull() {
#pragma omp parallel for schedule(static, 1024)
  for (int i = 0; i < nrElements; ++i) {
    elements[i].updateFull();
  }
}

void Mesh::rebuildCurrentEdgeSet() {
  std::unordered_map<EdgeKey, int, EdgeKeyHash> counts;
  counts.reserve(static_cast<size_t>(nrElements) * 3);
  for (const auto &e : elements) {
    const int id0 = e.ghostNodes[0].referenceId.i;
    const int id1 = e.ghostNodes[1].referenceId.i;
    const int id2 = e.ghostNodes[2].referenceId.i;
    ++counts[EdgeKey(id0, id1)];
    ++counts[EdgeKey(id1, id2)];
    ++counts[EdgeKey(id2, id0)];
  }
  currentEdgeSet.clear();
  currentEdgeSet.reserve(counts.size());
  for (const auto &kv : counts) {
    if (kv.second >= 2) {
      currentEdgeSet.insert(kv.first);
    }
  }
}

void Mesh::initializeEdgeSets() {
  rebuildCurrentEdgeSet();
  previousStepEdgeSet = currentEdgeSet;
  edgeFlipsSinceLastStep = 0;
}

void Mesh::updateEdgeSetForFlip(
    const std::array<const GhostNode *, 4> &nodeOrder, bool useMajorDiagonal) {
  if (currentEdgeSet.empty()) {
    throw std::runtime_error("updateEdgeSetForFlip: edge sets not initialized");
  }
  const EdgeKey diagMajor(nodeOrder[1]->referenceId.i,
                          nodeOrder[2]->referenceId.i);
  const EdgeKey diagMinor(nodeOrder[0]->referenceId.i,
                          nodeOrder[3]->referenceId.i);

  const EdgeKey &oldDiag = useMajorDiagonal ? diagMinor : diagMajor;
  const EdgeKey &newDiag = useMajorDiagonal ? diagMajor : diagMinor;

  currentEdgeSet.erase(oldDiag);
  currentEdgeSet.insert(newDiag);
}

std::size_t Mesh::countEdgeSetDelta(const EdgeSet &lhs,
                                    const EdgeSet &rhs) const {
  std::size_t count = 0;
  for (const auto &e : lhs) {
    if (rhs.find(e) == rhs.end()) {
      ++count;
    }
  }
  for (const auto &e : rhs) {
    if (lhs.find(e) == lhs.end()) {
      ++count;
    }
  }
  return count;
}

void Mesh::updateNodeForce(Node &node) {
  node.resetForce();
  for (size_t e = 0; e < node.elementCount; ++e) {
    const GhostNode *gn = node.connectedGhostNodes[e];
    assert(gn != nullptr);
    node.f += gn->f;
  }
}

void Mesh::applyForceFromElementsToNodes() {
  // Loop over all the nodes
  // (Looping over the elements would create a problem since two threads might
  // write to the same node at the same time. By looping over the nodes, each
  // thread only writes to one node at a time. There is no problem if two
  // threads read from the same element at the same time.)
#pragma omp parallel for
  for (int i = 0; i < (int)nodes.size(); ++i) {
    Node &n = nodes(i);
    updateNodeForce(n);
  }
}

void Mesh::rebuildConnectivity() {
  for (int i = 0; i < nodes.size(); ++i) {
    Node &n = nodes(i);
    n.elementCount = 0;
    n.connectedElements.fill(-1);
    n.nodeIndexInElement.fill(-1);
    n.connectedGhostNodes.fill(nullptr);
  }

  for (int eIdx = 0; eIdx < elements.size(); ++eIdx) {
    const TElement &e = elements[eIdx];
    for (int i = 0; i < 3; ++i) {
      const GhostNode &gn = e.ghostNodes[i];
      Node *node = (*this)[gn.referenceId];
      int &count = node->elementCount;
      if (count < MAX_ELEMENTS_PER_NODE) {
        node->connectedElements[count] = eIdx;
        node->nodeIndexInElement[count] = i;
        node->connectedGhostNodes[count] = nullptr;
        ++count;
      } else {
        throw std::overflow_error("Element index overflow for node " +
                                  std::to_string(gn.referenceId.i));
      }
    }
  }

  rebuildConnectedGhostNodes();
}

void Mesh::rebuildConnectedGhostNodes() {
  for (int i = 0; i < nodes.size(); ++i) {
    Node &n = nodes(i);
    for (int e = 0; e < n.elementCount; ++e) {
      const int elIdx = n.connectedElements[e];
      const int nodeIdx = n.nodeIndexInElement[e];
      if (elIdx >= 0 && elIdx < (int)elements.size() && nodeIdx >= 0 &&
          nodeIdx < 3) {
        n.connectedGhostNodes[e] = &elements[elIdx].ghostNodes[nodeIdx];
      } else {
        n.connectedGhostNodes[e] = nullptr;
      }
    }
    for (int e = n.elementCount; e < MAX_ELEMENTS_PER_NODE; ++e) {
      n.connectedGhostNodes[e] = nullptr;
    }
  }
}

void Mesh::updateConnectedGhostNodesForElement(int elementIndex) {
  if (elementIndex < 0 || elementIndex >= (int)elements.size()) {
    return;
  }
  TElement &e = elements[elementIndex];
  for (int i = 0; i < 3; ++i) {
    const GhostNode &gn = e.ghostNodes[i];
    Node *n = (*this)[gn.referenceId];
    for (int k = 0; k < n->elementCount; ++k) {
      if (n->connectedElements[k] == elementIndex &&
          n->nodeIndexInElement[k] == i) {
        n->connectedGhostNodes[k] = &gn;
        break;
      }
    }
  }
}

void Mesh::checkForces(std::vector<Node *> nodes) {
  for (Node *n : nodes) {
    updateNodeForce(*n);
    if (n->f.norm() > 1e-6) {
      std::cout << "Invalid: " << n->id << ": \t" << n->f << '\n';
      std::cout << n->f.norm() << '\n';
    } else {
      std::cout << "Valid: " << n->id << ": \t" << n->f << '\n';
      std::cout << n->f.norm() << '\n';
    }
  }
}

void Mesh::checkForces(const std::vector<GhostNode> nodes) {

  checkForces({(*this)[nodes[0].referenceId], (*this)[nodes[1].referenceId],
               (*this)[nodes[2].referenceId], (*this)[nodes[3].referenceId]});
}

inline bool inRegion(const Eigen::Matrix2d &G) {
  const double a = G(0, 0);
  const double b = G(0, 1);
  assert(G(0, 1) == G(1, 0));
  const double c = G(1, 1);
  if (b < 0) {
    return false;
  }
  if (b > std::min(a, c)) {
    return false;
  }
  return true;
}

inline bool inExtendedRegion(const Eigen::Matrix2d &G) {
  const double a = G(0, 0);
  const double b = 0.5 * (G(0, 1) + G(1, 0)); // robust off-diagonal
  const double c = G(1, 1);

  if (a <= c) {
    const double b_lo = -0.5 * a;
    const double b_hi = std::min(1.5 * a, (3.0 * a + c) / 4.0);
    return (b >= b_lo) && (b <= b_hi);
  } else {
    const double b_lo = -0.5 * c;
    const double b_hi = std::min(1.5 * c, (a + 3.0 * c) / 4.0);
    return (b >= b_lo) && (b <= b_hi);
  }
}

inline double maxCosineForMinAngle(const TElement &e) {
  // We want to compare the smalest angle, but
  // avoid expensive trigonometric functions.
  double maxCos = -1.0;
  for (int i = 0; i < 3; ++i) {
    const int next = (i + 1) % 3;
    const int prev = (i + 2) % 3;
    const Vector2d v1 = e.ghostNodes[next].pos - e.ghostNodes[i].pos;
    const Vector2d v2 = e.ghostNodes[prev].pos - e.ghostNodes[i].pos;
    const double denom = v1.norm() * v2.norm();
    if (denom <= 1e-12) {
      return 1.0;
    }
    double cosA = v1.dot(v2) / denom;
    cosA = std::clamp(cosA, -1.0, 1.0);
    if (cosA > maxCos) {
      maxCos = cosA;
    }
  }
  return maxCos;
}

// This function goes through each pair of elements and checks if the pair
// should flip their diagonal
void Mesh::resetFlipTracking() {
  if (flipRecords.size() < elements.size() / 2) {
    flipRecords.resize(elements.size() / 2);
  }
  flipRecordCount = 0;
  if (flippedThisReconnect.size() != elements.size()) {
    flippedThisReconnect.resize(elements.size());
  }
  std::fill(flippedThisReconnect.begin(), flippedThisReconnect.end(), 0);
}

void Mesh::undoReconnections() {
  if (flipRecordCount == 0) {
    return;
  }
  for (int idx = 0; idx < flipRecordCount; ++idx) {
    const FlipRecord &rec = flipRecords[idx];
    const int nElements = static_cast<int>(elements.size());
    if (rec.e1 < 0 || rec.e2 < 0 || rec.e1 >= nElements ||
        rec.e2 >= nElements) {
      continue;
    }
    std::array<const GhostNode *, 4> nodes = {nullptr, nullptr, nullptr,
                                              nullptr};
    for (int k = 0; k < 4; ++k) {
      nodes[k] = findGhostInPairById(rec.e1, rec.e2, rec.nodeIds[k]);
      if (!nodes[k]) {
        throw std::runtime_error("undoReconnections: missing ghost node id");
      }
    }
    // Undo by using the opposite diagonal for the same node ordering.
    flipEdge(rec.e1, rec.e2, nodes, false);
  }
  flipRecordCount = 0;
  if (!flippedThisReconnect.empty()) {
    std::fill(flippedThisReconnect.begin(), flippedThisReconnect.end(), 0);
  }
  markDirty();
}

bool Mesh::reconnect(bool onlyCheck) {
  ensureGeometry();
  if (!onlyCheck) {
    resetFlipTracking();
  }
  int nrBadChanges = 0;
  int nrGoodChanges = 0;
  meshReconnected = false;
  auto alreadyFlipped = [this](int idx) {
    return idx >= 0 && idx < static_cast<int>(flippedThisReconnect.size()) &&
           flippedThisReconnect[idx] != 0;
  };
  for (int i = 0; i < elements.size(); i++) {
    if (alreadyFlipped(i)) {
      continue;
    }

    TElement &e = elements[i];

    // Check if the geometry of the element is bad
    if (!inRegion(e.G)) {
      int twinIndex = e.getElementTwin(*(this));
      // If we found a twin
      if (twinIndex != -1) {
        TElement &twin = elements[twinIndex];
        if (alreadyFlipped(twinIndex)) {
          continue;
        }
        // Check if the twin element is also outside the triangular elastic
        // regime
        if (!inRegion(twin.G)) {
          // Both elements are outside the triangular elastic regime

          if (onlyCheck) {
            return true;
          }
          // Current smallest angle (via cosine proxy)
          const double oldMaxCos =
              std::max(maxCosineForMinAngle(e), maxCosineForMinAngle(twin));

          fixElementPair(e, twin);

          // New smallest angle (via cosine proxy)
          const double newMaxCos =
              std::max(maxCosineForMinAngle(e), maxCosineForMinAngle(twin));
          // We only accept the change if the smallest angle is improved
          if (newMaxCos > oldMaxCos) {
            nrBadChanges++;
            std::string suffix = "badFlip_e" + std::to_string(e.eIndex) + "_t" +
                                 std::to_string(twinIndex);
            // Capture the bad (post-flip) state.
            markDirty();
            writeToVtu("", true, VtuFieldLevel::All, suffix + "_after");

            // Revert just this flip using the last recorded pair.
            if (flipRecordCount <= 0) {
              throw std::runtime_error(
                  "Bad reconnection detected but no flip record available.");
            }
            const FlipRecord &rec = flipRecords[flipRecordCount - 1];
            std::array<const GhostNode *, 4> nodes = {nullptr, nullptr, nullptr,
                                                      nullptr};
            for (int k = 0; k < 4; ++k) {
              nodes[k] = findGhostInPairById(rec.e1, rec.e2, rec.nodeIds[k]);
              if (!nodes[k]) {
                throw std::runtime_error(
                    "Bad reconnection: missing ghost node id while reverting.");
              }
            }
            flipEdge(rec.e1, rec.e2, nodes, false);
            if (rec.e1 >= 0 &&
                rec.e1 < static_cast<int>(flippedThisReconnect.size())) {
              flippedThisReconnect[rec.e1] = 0;
            }
            if (rec.e2 >= 0 &&
                rec.e2 < static_cast<int>(flippedThisReconnect.size())) {
              flippedThisReconnect[rec.e2] = 0;
            }
            if (flipRecordCount > 0) {
              --flipRecordCount;
            }

            // Capture the reverted (pre-flip) state.
            markDirty();
            writeToVtu("", true, VtuFieldLevel::All, suffix + "_reverted");

            throw std::runtime_error(
                "Bad reconnection: flip worsened minimum angle (oldMaxCos=" +
                std::to_string(oldMaxCos) +
                ", newMaxCos=" + std::to_string(newMaxCos) + ").");
          } else {
            // std::cout << "Change was good! " << oldSmalestAngle << " to "
            //           << newSmallestAngle << '\n';
            nrGoodChanges++;
          }

          meshReconnected = true;
        } else {
          // The second twin element is not outside the allowed region,
          // so we don't flip the edge
          continue;
        }
      } else {
        // std::cerr << "Error: No twin found for element " << e.eIndex <<
        // ".\n";
        //  This will happen often, since the twin element checks if it's
        // largest angle is facing this element. If not, it will not be
        // considered a twin. (this might be changed in the future)
        // throw std::runtime_error("No twin found for element.");
      }
    }
  }
  if (nrBadChanges > 0) {
    std::cout << "Warning: " << nrBadChanges << " bad reconnections and "
              << nrGoodChanges << " good reconnections.\n";
  }
  if (!onlyCheck && meshReconnected) {
    markDirty();
  }
  return meshReconnected;
}

// Helpers

#if !defined(IDE_LIGHTWEIGHT)
inline Eigen::Vector2d toEigen(const Point &p) {
  return {CGAL::to_double(p.x()), CGAL::to_double(p.y())};
}
// Unique key for a triangle based on its three reference vertex indices
struct TriKey {
  uint64_t a, b, c;
  bool operator==(const TriKey &o) const noexcept {
    return a == o.a && b == o.b && c == o.c;
  }
  // Define less-than operator for use in std::set
  bool operator<(const TriKey &o) const noexcept {
    if (a != o.a)
      return a < o.a;
    if (b != o.b)
      return b < o.b;
    return c < o.c;
  }
};

static inline TriKey makeTriKey(const DelaunayInfo::Face_handle &f) {
  std::array<uint64_t, 3> v{0, 0, 0};
  for (int k = 0; k < 3; ++k) {
    v[k] = f->vertex(k)->info().refNodeIndex;
  }
  std::sort(v.begin(), v.end());
  return TriKey{v[0], v[1], v[2]};
}

void Mesh::reconnectDelaunay() {
  // We will refer to the native MTS mesh as the "MTSMesh", and the Delaunay
  // triangulation as the "Dmesh".
  // In order to deal with periodic boundaries, we will extend the mesh
  // by three node layers in each direction.
  int extension = 3;
  int extendedRows = rows + extension * 2;
  int extendedCols = cols + extension * 2;
  int nrNodesInDMesh = extendedRows * extendedCols;
  // Note that the same NodeId will appear multiple times
  std::vector<NodeId> DNodeToMTSNode; // index in Dmesh -> NodeId in MTSMesh
  DNodeToMTSNode.reserve(nrNodesInDMesh);

  // 2) Build Delaunay with vertex info = VInfo (baseId, dr, dc)
  DelaunayInfo dt;
  for (int dr = -extension; dr < rows + extension; ++dr) {
    for (int dc = -extension; dc < cols + extension; ++dc) {
      const GhostNode gn = getGhostNode(Vector2i{dc, dr});
      const Point p(gn.pos.x(), gn.pos.y());
      auto vh = dt.insert(p);
      VInfo vi;
      vi.refNodeIndex = gn.referenceId.i;
      vh->info() = vi;
    }
  }

  // 3) Remove excess elements
  // We only keep faces with two or more vertices inside the window
  // And we also make sure to only keep unique triangles (no duplicates)
  std::set<TriKey> seenTriangles;

  auto keep_element = [&](const DelaunayInfo::Face_handle &f) {
    // Deduplicate by canonical triangle key based on reference vertex indices
    TriKey key = makeTriKey(f);
    if (seenTriangles.find(key) != seenTriangles.end()) {
      return false; // duplicate triangle
    }

    const Point &p0 = f->vertex(0)->point();
    const Point &p1 = f->vertex(1)->point();
    const Point &p2 = f->vertex(2)->point();

    // For some reason, CGAL makes flat triagnles sometimes.
    const K::FT area = CGAL::area(p0, p1, p2); // non-negative area in K::FT
    if (area < K::FT(1e-5)) {
      return false; // near-flat triangle
    }

    // Half-open base window test via exact comparisons on K::FT
    const Point c = CGAL::centroid(p0, p1, p2);
    const double cx = CGAL::to_double(c.x());
    const double cy = CGAL::to_double(c.y());

    // We keep an extra row and column if using PBC. These are the elements
    // that wrap around.
    int extra = usingPBC ? 1 : 0;
    // Keep iff 0 <= cx < Lx and 0 <= cy < Ly (half-open [0,L))
    if (!(0 <= cx && cx < (cols - 1 + extra) * a && //
          0 <= cy && cy < (rows - 1 + extra) * a)) {
      return false;
    }

    seenTriangles.insert(key);
    return true;
  };

  // Collect faces to keep
  std::vector<DelaunayInfo::Face_handle> kept_faces;
  kept_faces.reserve(
      std::distance(dt.finite_faces_begin(), dt.finite_faces_end()));
  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
    if (keep_element(f)) {
      kept_faces.push_back(f);
    }
  }

  // 4) Sanity: number of faces should match our preallocated elements
  const int nFaces = static_cast<int>(kept_faces.size());
  if (nFaces != nrElements) {
    elements.resize(nFaces);
    std::cerr
        << "Warning: reconnectDelaunay(): triangle count mismatch (expected "
        << nrElements << ", got " << nFaces << ")." << std::endl;
    nrElements = nFaces;
  }

  // 5) Reset per-node connectivity
  for (long i = 0; i < nodes.size(); ++i) {
    nodes(i).elementCount = 0;
  }

  // 6) Refill elements from CGAL faces (each face -> one TElement)
  int eIdx = 0;
  for (const auto &f : kept_faces) {
    std::array<GhostNode, 3> g;
    for (int k = 0; k < 3; ++k) {
      auto v = f->vertex(k);
      const VInfo &vi = v->info();
      const Node *n = &nodes(vi.refNodeIndex); // reference node in MTS mesh
      // Build ghost from DT vertex coordinate. Use the targetPos to set the
      // periodic shift.
      g[k] = m_gn(n, toEigen(v->point()));
    }

    double noise = (eIdx < (int)elements.size()) ? elements[eIdx].noise
                                                 : sampleNormal(1, QDSD);
    elements[eIdx] = TElement((*this), g[0], g[1], g[2], eIdx, noise,
                              energyFunction, bulkModulus);
    ++eIdx;
  }

  // 7) Clear reconnection flags for the new element set
  resetFlipTracking();

  rebuildCurrentEdgeSet();
  rebuildConnectedGhostNodes();
  markDirty();
}
#else
void Mesh::reconnectDelaunay() {
  throw std::runtime_error(
      "reconnectDelaunay requires CGAL; build without IDE_LIGHTWEIGHT.");
}
#endif

std::array<GhostNode, 4> Mesh::getElementPairNodes(const TElement &e1,
                                                   const TElement &e2) {
  // We have two elements that share two nodes. Let's call them common nodes,
  // and the two other nodes angle nodes. We want to return the sequence:
  // {angleE1, coNode1, coNode2, angleE2}

  // we need to find which node is not in the other element
  const GhostNode *el1AngleNode = nullptr;
  const GhostNode *coNode1 = nullptr;
  const GhostNode *coNode2 = nullptr;
  for (int i = 0; i < 3; i++) {
    if (e1.ghostNodes[i].id != e2.ghostNodes[0].id &&
        e1.ghostNodes[i].id != e2.ghostNodes[1].id &&
        e1.ghostNodes[i].id != e2.ghostNodes[2].id) {
      el1AngleNode = &e1.ghostNodes[i];
      // We now assume that the two others are common nodes
      coNode1 = &e1.ghostNodes[(i + 1) % 3];
      coNode2 = &e1.ghostNodes[(i + 2) % 3];
      // If the id of coNode1 is larger than coNode2, we swap them
      if (coNode1->referenceId.i > coNode2->referenceId.i) {
        std::swap(coNode1, coNode2);
      }
      break;
    }
  }
  // Find the element in e2 that is not in e1.
  const GhostNode *el2AngleNode = nullptr;
  for (int i = 0; i < 3; i++) {
    if (e2.ghostNodes[i].id != e1.ghostNodes[0].id &&
        e2.ghostNodes[i].id != e1.ghostNodes[1].id &&
        e2.ghostNodes[i].id != e1.ghostNodes[2].id) {
      el2AngleNode = &e2.ghostNodes[i];
      break;
    }
  }
  const std::array<GhostNode, 4> assumedOrder = {*el1AngleNode, *coNode1,
                                                 *coNode2, *el2AngleNode};
  return assumedOrder;
  // // Note that this function only works for elements where the angle nodes
  // are
  // // "seeing" each other.
  // //  Extract the nodes with large angles from each element
  // const GhostNode &el1AngleNode = e1.ghostNodes[e1.angleNode];
  // const GhostNode &el2AngleNode = e2.ghostNodes[e2.angleNode];

  // // Extract the other nodes that are common between the elements
  // auto coAngleNodes = e1.getCoAngleNodes();

  // // Arrange the nodes in the required order for createElementPair:
  // // {angle0, coNode1, coNode2, angle3}:
  // // c2____a3
  // //  |  \  |
  // // a0____c1
  // // With angle nodes in positions 0 and 3
  // const std::vector<GhostNode> assumedOrder = {el1AngleNode,
  // *coAngleNodes[0],
  //                                              *coAngleNodes[1],
  //                                              el2AngleNode};

  // // Confirm that we have the same coAngleNodes
  // assert(coAngleNodes[0]->id == e2.getCoAngleNodes()[0]->id);
  // assert(coAngleNodes[1]->id == e2.getCoAngleNodes()[1]->id);

  // return assumedOrder;
}

std::vector<GhostNode>
Mesh::getUniqueNodes(const std::vector<TElement *> elements) {
  auto compById = [](const GhostNode &a, const GhostNode &b) {
    if (a.id.x() != b.id.x())
      return a.id.x() < b.id.x();
    return a.id.y() < b.id.y();
  };
  std::set<GhostNode, decltype(compById)> uniqueNodes(compById);

  for (const auto &e : elements) {
    for (const auto &gn : e->ghostNodes) {
      uniqueNodes.insert(gn);
    }
  }

  return {uniqueNodes.begin(), uniqueNodes.end()};
}

const GhostNode *Mesh::findGhostInPairById(int e1i, int e2i,
                                           const Vector2i &id) const {
  const TElement &e1 = elements[e1i];
  const TElement &e2 = elements[e2i];
  for (const auto &gn : e1.ghostNodes) {
    if (gn.id == id) {
      return &gn;
    }
  }
  for (const auto &gn : e2.ghostNodes) {
    if (gn.id == id) {
      return &gn;
    }
  }
  return nullptr;
}

void Mesh::recordFlipPair(int e1i, int e2i,
                          const std::array<const GhostNode *, 4> &nodeOrder) {
  if (flipRecordCount >= static_cast<int>(flipRecords.size())) {
    return;
  }
  FlipRecord &rec = flipRecords[flipRecordCount++];
  rec.e1 = e1i;
  rec.e2 = e2i;
  if (e1i >= 0 && e1i < static_cast<int>(flippedThisReconnect.size())) {
    flippedThisReconnect[e1i] = 1;
  }
  if (e2i >= 0 && e2i < static_cast<int>(flippedThisReconnect.size())) {
    flippedThisReconnect[e2i] = 1;
  }
  for (int k = 0; k < 4; ++k) {
    rec.nodeIds[k] = nodeOrder[k]->id;
  }
}

void Mesh::removeElementsFromNodes(
    const std::array<const GhostNode *, 4> &gNodes, int e1i, int e2i) {
  const std::array<Node *, 4> nodes = {
      (*this)[gNodes[0]->referenceId], (*this)[gNodes[1]->referenceId],
      (*this)[gNodes[2]->referenceId], (*this)[gNodes[3]->referenceId]};
  removeElementsFromNodes(nodes, e1i, e2i);
}

void Mesh::removeElementsFromNodes(const std::array<Node *, 4> &nodes, int e1i,
                                   int e2i) {
  for (Node *n : nodes) {
    int tempElementIndices[MAX_ELEMENTS_PER_NODE];
    int tempNodeIndexInElement[MAX_ELEMENTS_PER_NODE];
    const GhostNode *tempGhostNodes[MAX_ELEMENTS_PER_NODE];
    int newCount = 0;

    for (int i = 0; i < n->elementCount; i++) {
      const int el = n->connectedElements[i];
      if (el == e1i || el == e2i) {
        continue;
      }
      tempElementIndices[newCount] = n->connectedElements[i];
      tempNodeIndexInElement[newCount] = n->nodeIndexInElement[i];
      tempGhostNodes[newCount] = n->connectedGhostNodes[i];
      newCount++;
    }

    for (int i = 0; i < newCount; i++) {
      n->connectedElements[i] = tempElementIndices[i];
      n->nodeIndexInElement[i] = tempNodeIndexInElement[i];
      n->connectedGhostNodes[i] = tempGhostNodes[i];
    }
    for (int i = newCount; i < n->elementCount; i++) {
      n->connectedElements[i] = -1;
      n->nodeIndexInElement[i] = -1;
      n->connectedGhostNodes[i] = nullptr;
    }
    n->elementCount = newCount;
  }
}

void Mesh::flipEdge(int e1i, int e2i,
                    const std::array<const GhostNode *, 4> &nodeOrder,
                    bool useMajorDiagonal, bool preserveNoise) {
  updateEdgeSetForFlip(nodeOrder, useMajorDiagonal);
  removeElementsFromNodes(nodeOrder, e1i, e2i);
  createElementPair(nodeOrder, e1i, e2i, useMajorDiagonal, preserveNoise);
  updateConnectedGhostNodesForElement(e1i);
  updateConnectedGhostNodesForElement(e2i);
  elements[e1i].updateAngleNode();
  elements[e2i].updateAngleNode();
}

bool Mesh::canonicalizeElementPairNodes(const TElement &e1, const TElement &e2,
                                        std::array<GhostNode, 4> &out) const {
  const GhostNode *angleE1 = nullptr;
  const GhostNode *angleE2 = nullptr;
  std::array<const GhostNode *, 2> sharedE1 = {nullptr, nullptr};
  std::array<NodeId, 2> sharedIds = {NodeId(), NodeId()};
  int sharedCount = 0;

  for (const auto &gn1 : e1.ghostNodes) {
    bool isShared = false;
    for (const auto &gn2 : e2.ghostNodes) {
      if (gn1.referenceId == gn2.referenceId) {
        isShared = true;
        bool alreadyRecorded = false;
        for (int k = 0; k < sharedCount; ++k) {
          if (sharedIds[k] == gn1.referenceId) {
            alreadyRecorded = true;
            break;
          }
        }
        if (!alreadyRecorded && sharedCount < 2) {
          sharedE1[sharedCount] = &gn1;
          sharedIds[sharedCount] = gn1.referenceId;
          ++sharedCount;
        }
        break;
      }
    }
    if (!isShared) {
      angleE1 = &gn1;
    }
  }

  if (sharedCount != 2 || angleE1 == nullptr) {
    return false;
  }

  for (const auto &gn2 : e2.ghostNodes) {
    bool isShared = false;
    for (int k = 0; k < sharedCount; ++k) {
      if (gn2.referenceId == sharedIds[k]) {
        isShared = true;
        break;
      }
    }
    if (!isShared) {
      angleE2 = &gn2;
    }
  }

  if (angleE2 == nullptr) {
    return false;
  }

  GhostNode shared0 = *sharedE1[0];
  GhostNode shared1 = *sharedE1[1];
  if (shared0.referenceId.i > shared1.referenceId.i) {
    std::swap(shared0, shared1);
  }

  const Vector2d midpoint = 0.5 * (shared0.pos + shared1.pos);

  const Node *refAngle1 = (*this)[angleE1->referenceId];
  const Node *refAngle2 = (*this)[angleE2->referenceId];
  GhostNode angle1(refAngle1, midpoint, rows, cols, latticeBasis,
                   currentDeformation);
  GhostNode angle2(refAngle2, midpoint, rows, cols, latticeBasis,
                   currentDeformation);

  out = {angle1, shared0, shared1, angle2};
  return true;
}

void Mesh::fixElementPair(TElement &e1, TElement &e2) {
  // This function takes two elements that should both have large angles, and
  // reconfigures the 4 nodes into two new elements that have smaller angles.

  std::array<GhostNode, 4> standardOrder;
  if (!canonicalizeElementPairNodes(e1, e2, standardOrder)) {
    // Legacy fallback: try to align periodic images, then use old ordering.
    auto e1Co = e1.getCoAngleNodes();
    auto e2Co = e2.getCoAngleNodes();
    if (e1Co[0]->id != e2Co[0]->id || e1Co[1]->id != e2Co[1]->id) {
      fixPeriodicElementPair(e1, e2);
    }
    standardOrder = getElementPairNodes(e1, e2);
  }

  // When we give these nodes to the createElementPair function, it is
  // important To consider the order in which we give them. The function will
  // interpret the list of nodes like this {angle0, coNode1, coNode2, angle3}:
  // c2____a3
  //  |  \  |
  // a0____c1
  // Assuming that we use major diagonal (we could use either so long as we
  // change the order we give the nodes in), we will make g2 and g3 be the new
  // corner nodes. That means we should make the angle nodes go in the first
  // and last possition. The order of the two others nodes does not matter,
  // but by convention, we want to make the node with the smaller index come
  // first

  // This new order will swich what nodes are angle nodes and coAngle nodes
  const std::array<const GhostNode *, 4> newPairOrder = {
      &standardOrder[1], &standardOrder[0], &standardOrder[3],
      &standardOrder[2]};
  recordFlipPair(e1.eIndex, e2.eIndex, newPairOrder);
  flipEdge(e1.eIndex, e2.eIndex, newPairOrder, true);
}

void Mesh::fixPeriodicElementPair(TElement &e1, TElement &e2) {
  // The situation here is like this: We would usually fix an element pair
  // that looks like this: c2____a3
  //  |  \  |
  // a0____c1
  // But now, this element pair is accros the periodic boundary, like this:
  //  c2____a3           c2
  //      \  |    ...    |  \
  //        c1          a0____c1
  // What we need to do now, is to move one of the elements to the other
  // element so they can be reconnected properly. We will do this by considering
  // which element is closest to the center of the system. We will then move
  // the other one.

  // We simply move the one that is furthest away from the center of the mesh

  double distE1 = (e1.getCom() - com).norm();
  double distE2 = (e2.getCom() - com).norm();
  if (distE1 > distE2) {
    moveElementToTwin(e1, e2);
  } else {
    moveElementToTwin(e2, e1);
  }
}

void Mesh::moveElementToTwin(TElement &elementToMove,
                             const TElement &fixedElement) {
  // In order to move the element, we just copy the coAngleNodes from the
  // fixed element and find the appropreate periodic shift for the final angle
  // node.

  // Get the coAngleNodes of the fixed element, as these are the ghost
  // nodes we should be using to create the new element.
  auto fixedCoAngleNodes = fixedElement.getCoAngleNodes();
  // This is the node that we should move by giving it a new periodic shift
  // (We are not actually moving the reference node, but only the ghost node).
  Node *refNode = (*this)[elementToMove.ghostNodes[elementToMove.angleNode]];

  // Use the average position of the fixed co-angle nodes to get a stable base
  // shift (uses deformation-aware nearest-image logic in GhostNode ctor).
  const Vector2d p1 = fixedCoAngleNodes[0]->pos;
  const Vector2d p2 = fixedCoAngleNodes[1]->pos;
  const Vector2d targetPos = 0.5 * (p1 + p2);
  const GhostNode baseNode(refNode, targetPos, rows, cols, latticeBasis,
                           currentDeformation);
  const Vector2i baseShift = baseNode.periodicShift;

  // Try shifts around the base image and choose the one that minimizes the
  // distance to the coAngleNodes.
  Vector2i minPShift = baseShift;
  double minDistance = std::numeric_limits<double>::max();

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      Vector2i shift = baseShift + Vector2i{i * cols, j * rows};
      GhostNode testNode(refNode, shift, cols, latticeBasis,
                         currentDeformation);
      const double distance =
          (testNode.pos - p1).squaredNorm() + (testNode.pos - p2).squaredNorm();
      if (std::isfinite(distance) && distance < minDistance) {
        minDistance = distance;
        minPShift = shift;
      }
    }
  }

  const GhostNode cornerNode =
      GhostNode(refNode, minPShift, cols, latticeBasis, currentDeformation);

  // We remove the element from all the nodes it is connected to.
  removeElementFromNodes(elementToMove);

  // overwrite the element
  elementToMove = TElement{(*this),
                           cornerNode,
                           *fixedCoAngleNodes[0],
                           *fixedCoAngleNodes[1],
                           elementToMove.eIndex,
                           elementToMove.noise,
                           energyFunction,
                           bulkModulus};
}

int Mesh::countConnectionsInGhostNode(const GhostNode &gn) {
  int connections = 0;

  // First we find the reference node
  Node *n = (*this)[gn.referenceId];

  // Then we need to go through each element it is connected to, and compare
  // with the ghost node in those elements to see if they are the same ghost
  // node as the one we are considering. We check if they are the same by
  // comparing their row, col AND periodicShift.

  for (int i = 0; i < n->elementCount; i++) {
    // Get one of the elements connected to the reference node.
    TElement &e = elements[n->connectedElements[i]];
    // Find the ghost node in that element that represents the reference node.
    GhostNode &gnInElement = e.ghostNodes[n->nodeIndexInElement[i]];
    // Check if this node is the same node as our input ghost node.
    // If it is, it means that our input ghost node is connected to this
    // element. (That means that the minimum number of connections will almost
    // always be 1, since we count the element that the input ghost node comes
    // from.)
    if (gnInElement.id == gn.id) {
      connections += 1;
    }
  }

  return connections;
}

void Mesh::setDiagonal(int row, int col, bool useMajorDiagonal) {
  // get the 4 ghost nodes of the selected section of the mesh
  const std::vector<GhostNode> ghosts = getSquareGhostNodes(row, col);
  // Get the indexes of the elements
  auto [e1i, e2i] = getElementIndices(row, col);

  // Now we need to remove e1i and e2i from the 4 nodes, since they will be
  // added back in the tElement constructor
  removeElementsFromNodes(row, col, {e1i, e2i});

  // Update elements, preserving existing noise values
  createElementPair(ghosts, e1i, e2i, useMajorDiagonal, true);

  rebuildConnectedGhostNodes();
}

void Mesh::removeElementFromNodes(const TElement &element) {

  std::vector<const GhostNode *> ghostNodesVector;
  ghostNodesVector.reserve(3);

  for (const auto &gn : element.ghostNodes) {
    ghostNodesVector.push_back(&gn);
  }

  removeElementsFromNodes(ghostNodesVector, {element.eIndex});
}

void Mesh::removeElementsFromNodes(int row, int col,
                                   const std::vector<int> elIndexToRemove) {
  std::vector<Node *> nodes = getSquareNodes(row, col);
  removeElementsFromNodes({nodes[0], nodes[1], nodes[2], nodes[3]},
                          elIndexToRemove);
}

void Mesh::removeElementsFromNodes(const std::vector<const GhostNode *> gNodes,
                                   const std::vector<int> elIndexToRemove) {
  std::vector<Node *> nodes;
  nodes.reserve(gNodes.size()); // Reserve space for efficiency

  std::transform(
      gNodes.begin(), gNodes.end(), std::back_inserter(nodes),
      [this](const GhostNode *g) { return (*this)[g->referenceId]; });

  removeElementsFromNodes(nodes, elIndexToRemove);
}

void Mesh::removeElementsFromNodes(std::vector<Node *> nodes,
                                   const std::vector<int> elIndexToRemove) {

  // We check each node
  for (Node *n : nodes) {
    std::vector<int> indexesToRemove;

    // Find indices of elements to remove
    for (int i = 0; i < n->elementCount; i++) {
      for (int j = 0; j < elIndexToRemove.size(); j++) {
        if (n->connectedElements[i] == elIndexToRemove[j]) {
          indexesToRemove.push_back(i);
          break; // Found a match, no need to check other elements
        }
      }
    }

    // If there's nothing to remove, continue to next node
    if (indexesToRemove.empty()) {
      continue;
    }

    // Create temporary arrays to store elements we want to keep
    int tempElementIndices[MAX_ELEMENTS_PER_NODE];
    int tempNodeIndexInElement[MAX_ELEMENTS_PER_NODE];
    const GhostNode *tempGhostNodes[MAX_ELEMENTS_PER_NODE];
    int newCount = 0;

    // Copy only the elements we want to keep
    for (int i = 0; i < n->elementCount; i++) {
      // Check if this index should be removed
      bool shouldRemove = false;
      for (int removeIdx : indexesToRemove) {
        if (i == removeIdx) {
          shouldRemove = true;
          break;
        }
      }

      // If not marked for removal, keep it
      if (!shouldRemove) {
        tempElementIndices[newCount] = n->connectedElements[i];
        tempNodeIndexInElement[newCount] = n->nodeIndexInElement[i];
        tempGhostNodes[newCount] = n->connectedGhostNodes[i];
        newCount++;
      }
    }

    // Update the node with the new arrays
    for (int i = 0; i < newCount; i++) {
      n->connectedElements[i] = tempElementIndices[i];
      n->nodeIndexInElement[i] = tempNodeIndexInElement[i];
      n->connectedGhostNodes[i] = tempGhostNodes[i];
    }
    for (int i = newCount; i < n->elementCount; i++) {
      n->connectedElements[i] = -1;
      n->nodeIndexInElement[i] = -1;
      n->connectedGhostNodes[i] = nullptr;
    }

    // Update the count
    n->elementCount = newCount;
  }
}

// Helper function to update positions using a generic buffer and its size
void Mesh::updateNodePositions(const double *data, size_t length) {
  const size_t nr_x_values = length / 2;
  const double *xData = data;
  const double *yData = data + nr_x_values;
  const size_t freeCount = freeNodeIds.size();
  for (size_t i = 0; i < freeCount; i++) {
    Node *n = (*this)[freeNodeIds[i]];
    n->setDisplacement({xData[i], yData[i]});
  }
  markDirty();
}

void Mesh::saveNodeDisplacements(std::vector<double> &displacements) const {
  const size_t nodeCount = static_cast<size_t>(rows * cols);
  displacements.resize(2 * nodeCount);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      const Node &n = nodes(row, col);
      const Vector2d &disp = n.u();
      const size_t idx = static_cast<size_t>(row * cols + col);
      displacements[idx] = disp[0];
      displacements[idx + nodeCount] = disp[1];
    }
  }
}

// Helper function to update positions using a generic buffer and its size
void Mesh::restoreNodeDisplacements(const std::vector<double> &displacements) {
  const size_t nodeCount = static_cast<size_t>(rows * cols);
  if (displacements.size() != 2 * nodeCount) {
    throw std::runtime_error(
        "Displacement buffer size does not match node count.");
  }
  const double *xData = displacements.data();
  const double *yData = displacements.data() + nodeCount;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      Node &n = nodes(row, col);
      const size_t idx = static_cast<size_t>(row * cols + col);
      n.setDisplacement({xData[idx], yData[idx]});
    }
  }
  markDirty();
  // We intentionally don't try to "rewind" history-style maxima (like
  // maxM3Nr/maxPlasticJump) to ensure they were achieved in the reverted mesh.
  // This keeps revert simple; it is only used when the mesh is already close
  // to equilibrium, so the impact should be negligible.
  updateMesh();
}

void Mesh::saveSnapshot(Snapshot &snapshot) const {
  snapshot.load = load;
  snapshot.loadSteps = loadSteps;
  snapshot.currentDeformation = currentDeformation;
  saveNodeDisplacements(snapshot.displacements);
}

Mesh::Snapshot Mesh::snapshotState() const {
  Snapshot snapshot;
  saveSnapshot(snapshot);
  return snapshot;
}

void Mesh::restoreState(const Snapshot &snapshot) {
  load = snapshot.load;
  loadSteps = snapshot.loadSteps;
  currentDeformation = snapshot.currentDeformation;
  restoreNodeDisplacements(snapshot.displacements);
}

double Mesh::rmsDistanceToSnapshot(const Snapshot &snapshot,
                                   bool subtractAvgU) const {
  const size_t freeCount = freeNodeIds.size();
  if (freeCount == 0) {
    return 0.0;
  }
  const size_t nodeCount = static_cast<size_t>(rows * cols);
  if (snapshot.displacements.size() != 2 * nodeCount) {
    throw std::runtime_error(
        "Snapshot displacements size does not match node count.");
  }

  const double *xSnapU = snapshot.displacements.data();
  const double *ySnapU = snapshot.displacements.data() + nodeCount;
  Vector2d avgUCurrent = Vector2d::Zero();
  Vector2d avgUSnap = Vector2d::Zero();
  if (subtractAvgU) {
    for (size_t i = 0; i < freeCount; i++) {
      const Node *n = (*this)[freeNodeIds[i]];
      const Vector2d u = n->u();
      const size_t idx = static_cast<size_t>(freeNodeIds[i].i);
      const Vector2d snapU = Vector2d{xSnapU[idx], ySnapU[idx]};
      avgUCurrent += u;
      avgUSnap += snapU;
    }
    avgUCurrent /= static_cast<double>(freeCount);
    avgUSnap /= static_cast<double>(freeCount);
  }

  double sum = 0.0;
  for (size_t i = 0; i < freeCount; i++) {
    const Node *n = (*this)[freeNodeIds[i]];
    Vector2d u = n->u();
    const size_t idx = static_cast<size_t>(freeNodeIds[i].i);
    Vector2d snapU = Vector2d{xSnapU[idx], ySnapU[idx]};
    if (subtractAvgU) {
      u -= avgUCurrent;
      snapU -= avgUSnap;
    }
    const Vector2d diff = u - snapU;
    sum += diff.squaredNorm();
  }
  return std::sqrt(sum / static_cast<double>(freeCount));
}

void Mesh::updateBoundingBox() {
  ensureForces();
  // Reset the bounding box
  bounds[0] = -100000; // max x
  bounds[1] = 100000;  // min x
  bounds[2] = -100000; // max y
  bounds[3] = 100000;  // min y

  // Update bounding box
  for (int i = 0; i < nrElements; i++) {
    for (int j = 0; j < 3; j++) {
      // Update bounds for x-coordinate
      if (elements[i].ghostNodes[j].pos[0] > bounds[0])
        bounds[0] = elements[i].ghostNodes[j].pos[0]; // max x
      if (elements[i].ghostNodes[j].pos[0] < bounds[1])
        bounds[1] = elements[i].ghostNodes[j].pos[0]; // min x

      // Update bounds for y-coordinate
      if (elements[i].ghostNodes[j].pos[1] > bounds[2])
        bounds[2] = elements[i].ghostNodes[j].pos[1]; // max y
      if (elements[i].ghostNodes[j].pos[1] < bounds[3])
        bounds[3] = elements[i].ghostNodes[j].pos[1]; // min y
    }
  }
}

void Mesh::updateAngles() {
  // It may be strange that updateAngles is not part of updateElements, but
  // angles are not used in the simulation, and relatively expensive to
  // calculate. They could be useful for analysis, so we do save them in the vtu
  // files.
  ensureForces();
  for (auto &el : elements) {
    el.updateAngles();
  }
}

void Mesh::updateAveragesAndPlasticEvents() {
  ensureFull();

  // Note that totalEnergy has already been calculated since we use it in
  // the energy minimization

  // we reset the maxEnergy
  maxEnergy = 0;

  // We calculate total force for debugging (should be zero)
  // Vector2d totalForce = Vector2d::Zero();
  // for (int i = 0; i < nodes.size(); i++) {
  //   totalForce += nodes(i).f;
  // }
  // // Mathematically, the total force should always be 0 (even out of
  // // equilibrium), but due to some rounding errors (i think), we need to
  // allow
  // // some freedom before we declare that something is wrong.
  // if (totalForce.norm() / nrNodes > 1e-13) {
  //   std::cout << "Min step: " << nrMinItterations << '\n';
  //   std::cout << "Total force: " << totalForce << '\n';
  //   std::cout << "Big force: " << totalForce.norm() << '\n';
  //   if (endOfStep) {

  //     std::cerr << "Total force is not zero. Something is wrong.";
  //   }
  // }

  // This is the total energy from all the triangles
  nr_elements_with_m3_fix_change = 0;
  nr_elements_with_m3_fix_changeInStep = 0;
  redQuadrantCounts = {0, 0, 0, 0};
  redQuadrantFixedCounts = {0, 0, 0, 0};
  double totalP11 = 0;
  double totalP12 = 0;
  double totalP21 = 0;
  double totalP22 = 0;
  double totalSigma11 = 0;
  double totalSigma12 = 0;
  double totalSigma22 = 0;
  double totalSigmaTrace = 0;
  sumM3Nr = 0;
  for (int i = 0; i < nrElements; i++) {
    const TElement &e = elements[i];
    totalP11 += e.P(0, 0);
    totalP12 += e.P(0, 1);
    totalP21 += e.P(1, 0);
    totalP22 += e.P(1, 1);
    totalSigma11 += e.sigma(0, 0);
    totalSigma12 += e.sigma(0, 1);
    totalSigma22 += e.sigma(1, 1);
    totalSigmaTrace += e.sigma.trace();
    if (e.red_quadrant >= 1 && e.red_quadrant <= 4) {
      redQuadrantCounts[static_cast<size_t>(e.red_quadrant - 1)] += 1;
    }
    if (e.red_quadrant_fixed >= 1 && e.red_quadrant_fixed <= 4) {
      redQuadrantFixedCounts[static_cast<size_t>(e.red_quadrant_fixed - 1)] +=
          1;
    }

    // We also keep track of the highest energy and some other things
    if (e.energy > maxEnergy) {
      maxEnergy = e.energy;
    }

    // Plastic event counters are based on pastM3Nr_fix/pastStepM3Nr_fix, which
    // are reset in resetPastPlasticCount.
    if (e.m3Nr > maxM3Nr) {
      maxM3Nr = e.m3Nr;
    }
    sumM3Nr += e.m3Nr;
    int plasticChange = e.m3Nr_fix - e.pastM3Nr_fix;
    if (plasticChange > maxPlasticJump) {
      maxPlasticJump = plasticChange;
    } else if (plasticChange < minPlasticJump) {
      minPlasticJump = plasticChange;
    }
    if (e.pastM3Nr_fix != e.m3Nr_fix) {
      nr_elements_with_m3_fix_change += 1;
    }
    if (e.pastStepM3Nr_fix != e.m3Nr_fix) {
      nr_elements_with_m3_fix_changeInStep += 1;
    }
  }

  averageEnergy = totalEnergy / nrElements;
  averageP11 = totalP11 / nrElements;
  averageP12 = totalP12 / nrElements;
  averageP21 = totalP21 / nrElements;
  averageP22 = totalP22 / nrElements;
  averageSigma11 = totalSigma11 / nrElements;
  averageSigma12 = totalSigma12 / nrElements;
  averageSigma22 = totalSigma22 / nrElements;
  averageSigmaTrace = totalSigmaTrace / nrElements;

  // We also calculate the number of edge flips that has occured
  const std::size_t changed =
      countEdgeSetDelta(currentEdgeSet, previousStepEdgeSet);
  edgeFlipsSinceLastStep = changed / 2;
}

void Mesh::updateCom() {
  ensureForces();
  com = Vector2d::Zero();
  for (int i = 0; i < nrElements; i++) {
    com += elements[i].getCom();
  }
  com /= nrElements;
}

// Moves a section of the mesh based on spatial coordinates (x, y).
void Mesh::moveMeshSection(double minX, double minY, Vector2d disp,
                           bool moveFixed, bool moveFree, double maxX,
                           double maxY) {
  auto isInBounds = [&](const Node &n) {
    const Vector2d &p = n.pos();
    return (p[0] >= minX && p[0] <= maxX && p[1] >= minY && p[1] <= maxY);
  };

  if (moveFixed) {
    for (const NodeId &nId : fixedNodeIds) {
      Node *n = (*this)[nId];
      if (isInBounds(*n)) {
        n->addDisplacement(disp);
      }
    }
  }

  if (moveFree) {
    for (const NodeId &nId : freeNodeIds) {
      Node *n = (*this)[nId];
      if (isInBounds(*n)) {
        n->addDisplacement(disp);
      }
    }
  }
  markDirty();
}

void Mesh::writeToVtu(std::string filename, bool minimizationStep) {
  writeToVtu(std::move(filename), minimizationStep, VtuFieldLevel::All, "");
}

void Mesh::writeToVtu(std::string filename, bool minimizationStep,
                      VtuFieldLevel level, std::string nameSuffix) {
  ensureFull();
  if (minimizationStep) {
    // We will save a lot of meshes, so these optimizations help quite a bit.
    writeMeshToVtu((*this), simName, dataPath, filename, minimizationStep,
                   level, nameSuffix);
  } else {
    // For video making, we need to have accurate bounding boxes.
    writeMeshToVtu((*this), simName, dataPath, filename, minimizationStep,
                   level, nameSuffix);
  }
}

void transform(const Matrix2d &matrix, Mesh &mesh,
               std::vector<NodeId> nodesToTransform) {
  // We get the adress of each node
  for (NodeId &nodeId : nodesToTransform) {
    transformInPlace(matrix, *mesh[nodeId]);
  }
  mesh.markDirty();
}

std::ostream &operator<<(std::ostream &os, const Mesh &mesh) {
  os << "Mesh: " << mesh.rows << " x " << mesh.cols << " nodes\nIds:\n";
  for (int i = mesh.cols - 1; i >= 0; --i) {
    for (int j = 0; j < mesh.rows; ++j) {
      Node n = mesh.nodes(i, j);
      os << n.id.i << "\t";
    }
    os << "\n";
  }
  os << "Positions:\n";
  for (int i = mesh.cols - 1; i >= 0; --i) {
    for (int j = 0; j < mesh.rows; ++j) {
      Node n = mesh.nodes(i, j);
      // set precision to 1
      os << std::fixed << std::setprecision(1) << n.pos().transpose() << "    ";
    }
    os << "\n";
  }

  return os;
}

void transform(const Matrix2d &matrix, Mesh &mesh) {
  // Transform all nodes
  transform(matrix, mesh, mesh.fixedNodeIds);
  transform(matrix, mesh, mesh.freeNodeIds);
}

void translate(Mesh &mesh, std::vector<NodeId> nodesToTranslate, double x,
               double y) {
  // We get the adress of each node
  for (NodeId &nodeId : nodesToTranslate) {
    translateInPlace(*mesh[nodeId], x, y);
  }
  mesh.markDirty();
}
void translate(Mesh &mesh, double x, double y) {
  translate(mesh, mesh.fixedNodeIds, x, y);
  translate(mesh, mesh.freeNodeIds, x, y);
}
