/**
 * BASIC 2D FEM on a 3D MESH
 *
 * This follows the method described in: 
 * a) Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 * b) Hans Peter Langtangen, "Introduction to finite element methods", 2014 
 *    http://hplgit.github.io/INF5620/doc/pub/H14/fem/html/main_fem.html
 * 
 * 
 * Solve Heat equation: "U dt = laplacian U” with Dirichlet boundary conditions
 * using Finite Element Method (FEM)
 * 
 * Laplacian U(x,y, t) = dU(x,y, t)/dt   in the region D
 *           U(x,y, t) = G(x,y)             on the region boundary #D for all times
 *           U(x,y, 0) = U_0(x,y)           initial condition at time t = 0
 * 
 * Since dU(x,y, t)/dt is a time derivative we discretize the time using Backguard Euler
 * 
 * dU(x,y, t)/dt = U(x,y, t + delta_t) - U(x,y, t) / delta_t
 * 
 * Replacing on original PDE:
 * 
 * U(x,y, t + delta_t) - U(x,y, t) / delta_t = Laplacian U(x,y, t + delta_t)    for backguard euler
 * 
 * Re-arranging:
 * 
 * U(x,y, t + delta_t) - delta_t * Laplacian U(x,y, t + delta_t) = U(x,y, t)
 * 
 * or better:
 * 
 * U(x,y,t) - delta_t * Laplacian U(x,y,t) = U(x,y, t_old)
 * 
 * The integral form (notice Integral_D is a double integral):
 * 
 * Integral_D U(x,y,t) - delta_t * Integral_D Laplacian U(x,y,t) dA ) = Integral_D ( U(x,y, t_old) dA )
 * 
 * 
 * WEAK FORM
 * 
 * The Variational Method. Here one tries to get a solution to the PDE by translating 
 * it into a minimization problem for an energy function that is a linear functional defined 
 * on a function space D. This involves to create an artificial functional for the problem. 
 * Solving the variational problem may require less continuity than that of the actual 
 * solution (hence WEAK form). 
 * 
 * Consider the weight function V(x,y) (a.k.a. test function) with compact support defined on D
 * 
 * minimize Integral_D V(x, y) R(x,y) dA = 0
 * 
 * Where R(x,y) is the residual or error of the function I.e., R(x,y) = laplacian U(x,y) - F(x,y)
 * The weak form of the PDE is
 * 
 * V(x, y) R(x,y) = 0
 * V(x, y) (U(x,y,t) - delta_t * Laplacian U(x,y,t) - U(x,y, t_old)) = 0
 * U(x,y,t) V(x, y) - delta_t * V(x, y) Laplacian U(x,y,t) = U(x,y, t_old)) V(x, y)
 * 
 * and the integral (variational) form of the PDE is
 * 
 * Integral_D U(x,y,t) V(x, y) dA - delta_t * Integral_D V(x, y) Laplacian U(x,y,t) dA = Integral_D U(x,y, t_old)) V(x, y) dA
 * 
 * NOTE on inner products: The integral form is more important since integral of the product 
 * of two functions f(x,y) and g(x,y) is the “inner product” integral_D( f(x,y) g(x,y) dA ).
 * If function g(x,y) is a “test function” then the inner product is way to “measure” the 
 * function f(x,y) on location where g(x,y) is defined. Since g(x,y) is not the Dirac Delta 
 * function then the measure is an averaged one.
 * 
 * Consider the vector field E(X) = V(X) grad U(X)
 * 
 * Taking div E(X) we get:
 * div E(X) = div (V(X) grad U(X)) = < grad V, grad U> + V laplacian U (by Green's first identity)
 * 
 * By applying the Divergence theorem:
 * 
 * Integral_D( div E(X) dA ) = Integral_#D ( <E, n> dS )  (where n is the normal to the boundary #D)
 * 
 * We get:
 * 
 * Integral_D( < grad V, grad U> + V laplacian U dA) = Integral_#D ( <V grad U, n> dS )
 *                                                   = Integral_#D ( V <grad U, n> dS )
 * 
 * So we can re-arrange the left-hand-side of our variational form of PDE as:
 * 
 * Integral_D(V laplacian U dA) = Integral_#D ( V <grad U, n> dS ) - Integral_D( < grad V, grad U> dA)
 * 
 * Replacing the left-hand-side of our variational form of PDE with the above we get:
 * 
 * Integral_D U V dA - delta_t * (Integral_#D ( V <grad U, n> dS ) - Integral_D( < grad V, grad U> dA)) = Integral_D U_old V dA
 * 
 * We now define that all V(x, y) at the boundary #D has zero value, so
 * 
 * V(x,y) = 0    for all (x,y) in #D
 * 
 * So the weak form is now:
 * 
 * Integral_D U V dA - delta_t * (- Integral_D( < grad V, grad U> dA)) = Integral_D U_old V dA
 * 
 * giving:
 * 
 * Integral_D U V dA + delta_t * Integral_D( < grad V, grad U> dA) = Integral_D U_old V dA
 * 
 * 
 * DISCRETIZATION
 * 
 * The domain D has to be meshed in N nodes and T triangles. For linear elements the Nodes coincide with 
 * mesh vertices (but for quadratic elements there will Nodes defined on the mid-point of edges of triangles,
 * so there can be more Nodes than vertices).
 * 
 * The first discretisation is easy. We discretize the test functions V(x, y) = K_i(x, y) for every node N 
 * to be equal to 0 or 1 and linear everywhere else:
 * 
 * K_j(N_i) = 1   if j = i 
 * K_j(N_i) = 0   if i != j
 * K_j(x, y) = linear between 0 and 1
 * 
 * So we get a system of integral equations, one equation per Node/test function K_i:
 * 
 * Integral_Di U K_i(x,y) dA + delta_t * Integral_Di( < grad K_i(x,y), grad U> dA) = Integral_Di U_old K_i(x,y) dA
 * 
 * Where the integrals are now per triangle-ring around node N_i or test function K_i
 * 
 * Now we define the global solution U(x,y) to be a finite sum:
 * 
 * U(x,y) = sum_j^N( U_j H_j(x,y) )
 * 
 * Where U_i are coefficients to be determined and H_j is called shape function (i.e., interpolation function) 
 * a function defined on mesh nodes as 0 or 1 and linear everywhere else (i.e., P1 linear functions):
 * 
 * H_j(N_i) = 1   if j = i 
 * H_j(N_i) = 0   if i != j
 * H_j(x,y) = linear between 0 and 1
 * 
 * In this case the test function K_i is the same as the shape function H_i, when that happens this is the 
 * so called Galerkin FEM. 
 * 
 * We discretise the < grad V, grad U> for each element Ti on the triangle-set around node N_i as contribution of nodes N_j:
 * 
 * grad U_i(x,y) = sum_i U_i grad H_i(x,y)
 * grad V_i(x,y) = grad K_i(x,y)
 * 
 * Plugging into: < grad V, grad U>
 * 
 * < grad V, grad U> = < grad K_i(x,y), sum_i U_i grad H_i(x,y)>
 *                   = sum_j < grad K_i(x,y), U_j grad H_j(x,y)> 
 *                   = sum_j U_j < grad K_i(x,y), grad H_j(x,y)> 
 * 
 * < grad V, grad U>_i = sum_j U_j <grad H_j, grad K_i>    (one equation with many unknowns U_j for each test function)
 * 
 * Integral_Di( < grad V_i, grad U_j> dA)  = sum_j U_j  Integral_Di( <grad H_j, grad K_i> dA)
 * 
 * So we get the following system of linear equations:
 * 
 * sum_j U_j  Integral_Di H_j K_i dA + delta_t * Integral_Di( < grad K_i, grad H_j> dA) = sum_j U_j_old Integral_Di H_j K_i dA
 * 
 * Which can be written as:
 * 
 * L_ij = Integral_Di( <grad H_j, grad K_i> dA
 * M_ij = Integral_Di H_j K_i dA
 * 
 * (M_ij + L_ij) U_i = M_ij U_i_old
 * 
 * Which can be assemble to form a global matrix system (we choose to have shape functions H_i = test functions
 * K_i so we have same number of equations than unknowns)
 * 
 * (M + L) U = M U_old
 * 
 * 
 * 
 * OPTIMIZATION (used in this program)
 * 
 * If you use a approximate quadrature rule for the mass matrix instead of the exact one an optimization is possible.
 * Exact quadrature rule for mass matrix (quadratic form) is evaluating the integral at mid-point of the triangle edges.
 * If instead we evaluate the integral at the vertices we get a linear approximation. The benefit is that the mass
 * matrix becomes a diagonal matrix where the diagonal values are 1/3 multiplied by the area of the face (constant). 
 * On each vertex that amounts to the sum of the areas of the sorrounding faces multiplied by 1/3. 
 * 
 * M_ii = 1/3 sum_ij area_of_face_j_indident_to_vertex_i
 * 
 * 
 * That diagonal matrix can be easily inverted by inverting the diagonal entries so we get:
 * 
 * M^-1 (M + L) U = M^-1 M U_old
 * 
 * which yields:
 *
 * (I + M^-1 L) U = U_old
 * 
 * M^-1 L = amounts to multiply the entire ith row of L with the ith diagonal value of M
 * 
 * This is convinient as there is no M at right-hand-side of the iteration
 * 
 * 
 * 
 * DEFINITION OF SHAPE AND TEST FUNCTIONS ON ELEMENTS
 * 
 * 
 * ASSEMBLY
 * 
 * The process of creating the matrix M_ij is called assembly. It consist of creating the matrix M^e_ij for each 
 * element (triangle) and then add it to the global M_ij matrix. 
 * 
 * The global matrix M_ij is NxN where N is number of mesh nodes. In theory the matrix M^e_ij is a NxN super-sparse 
 * version of M_ij just with a few non-zero entries which corresponds to a single element. So
 * 
 * M_ij = sum_e M^e_ij
 * 
 * In practice the M^e_ij is not NxN super-sparse matrix, but a 3x3 matrix (in case of linear triangle elements).
 * there is a map from indices from the 3x3 matrix to the NxN matrix as follows.
 * 
 * Each triangle's nodes has a global index assigned in the mesh, we also establish a local indexing of triangle
 * like 1, 2 and 3. We define a trivial bijective map between global and local indices.
 * 
 * 
 * 
 * REFERENCE ELEMENTS
 * 
 * One of the novelties of FEM is to be able to compute the integrals per individual elements on an easy and generic 
 * way by using so called Reference Elements.
 * 
 * For triangle elements we just need a single reference triangle
 * 
 * P3
 * º
 * |   \
 * |        \
 *               \
 * º---------------º
 * P1              P2
 * 
 * P1 = (0,0)
 * P2 = (1,0)
 * P3 = (0,1)
 * 
 * The P1 (linear) Lagrange shape functions N(u,v) has the form 
 * 
 * N(u,v) = A u + B u + C
 * 
 * For the reference trignale defined above the N(u,v) takes the simple form:
 * 
 * N_1(u,v) = 1 - u - v
 * N_2(u,v) = u
 * N_3(u,v) = v
 * 
 * We can check that <N_i, N_j> = Kronecker-Delta_ij 
 * 
 * N_1(P1) = 1   N_1(P2) = 0    N_1(P3) = 0
 * N_2(P1) = 0   N_2(P2) = 1    N_2(P3) = 0
 * N_3(P1) = 0   N_1(P2) = 0    N_1(P3) = 1
 * 
 * We define a function F that maps reference triangle to "physical" triangle
 * (the inverse F^-1 maps triangle from physical to reference frame)
 * 
 * | x |  = | x_2 - x_1   x_3 - x_1 | | u | + | x_1 |
 * | y |    | y_2 - y_1   y_3 - y_1 | | v |   | y_1 |
 * 
 *  X     =  B                         U    + X_1
 * 
 * F(U)       = B U + X_1          In K
 * F^-1(X)  = B^-1(X - X_1)        In Ref 
 * 
 * We denote N^k_i the function N_i in defined on physical triangle K
 * 
 * N^k_i(X) = N_i º F^-1(X)
 * N^k_i(X) = N_i( F^-1(X) ) 
 * 
 * The gradient of N^k_i(X) in physical triangle is:
 * 
 * grad_X N^k_i(X) = grad_X N_i( F^-1(X) )
 *                 = grad_U N_i( U ) * grad_X F^-1(X)    By chain rule
 *                 = B^-T grad_U N_i( U )
 * 
 * The gradient of N_i(U) in reference triangle is:
 * 
 * grad_U N_1(U) = (-1, -1)
 * grad_U N_2(U) = (1, 0)
 * grad_U N_3(U) = (0, 1)
 * 
 * The gradient of the inverse map F^-1(X) is:
 * 
 * grad_X F^-1(X) = grad_X B^-1(X - X_1) 
 *                = grad_X B^-1 X - grad B^-1 X_1     (but grad B^-1 X_1 = 0 as there is no X)
 *                = grad_X B^-1 X
 *                = B^-T      (transposed)
 * 
 * So the gradient of N^k_i(X) in physical triangle is "constant":
 * 
 * grad_X N^k_1(X) = B^-T (-1, -1)
 * grad_X N^k_2(X) = B^-T (1, 0)
 * grad_X N^k_3(X) = B^-T (0, 1)
 * 
 * 
 * 
 * NUMERICAL QUADRATURE
 * 
 * The entries of the matrix w_ij are the following double integrals:
 * 
 * w_ij = integral_k <grad N^k_i(X), grad N^k_j(X)> dA
 * 
 * Let thefine the inner product of basis functions as  I^k_ij(X) = <grad N^k_i(X), grad N^k_j(X)>
 * 
 * The 3-point Gauss quadrature rule for the linear triangle is:
 * 
 * w_ij = area K / 3 ( I^k_ij(X_1) +  I^k_ij(X_2) +  I^k_ij(X_3))
 * 
 * evaluated at the vertices of the triangle X1, X2 and X3. Which is an exact quadrature formula 
 * for linear function on a triangle. However since grad N^k_i(X) does not depends on X the quadrature 
 * expression reduces to:
 * 
 * w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
 * 
 * 
 * 
 * FOR 3D TRIANGLES
 * 
 * For 3D triangles the map function F changes as follows:
 * 
 * | x |  = | x_2 - x_1   x_3 - x_1  |   | u | + | x_1 |
 * | y |    | y_2 - y_1   y_3 - y_1  |   | v |   | y_1 |
 * | z |    | z_2 - z_1   z_3 - z_1  |           | z_1 |
 * 3x1      3x2                           2x1    3x1
 * 
 * F(U) = B U + X_1                         In K
 * F^-1(X) = (B^T B)^-1 B^T (X - X_1)       In Ref 
 * 
 * or
 * 
 * F^-1(X) = B^-* (X - X_1)                 where B^-* = (B^T B)^-1 B^T 
 * 
 * Note: B^-* is the pseudo-inverse of B
 * 
 * The same analysis as before holds so:
 * 
 * grad_X N^k_1(X) = B^-*T (-1, -1)
 * grad_X N^k_2(X) = B^-*T (1, 0)
 * grad_X N^k_3(X) = B^-*T (0, 1)
 * 
 */
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif

#if defined (__APPLE__) || defined (OSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include "GA/c3ga.h"
#include "GA/c3ga_util.h"
#include "GA/gl_util.h"

#include "primitivedraw.h"
#include "gahelper.h"
#include "Laplacian.h"

#include <memory>

#include <vector>
#include <queue>
#include <map>
#include <fstream>
#include <functional>
#include "numerics.h"
#include "HalfEdge/Mesh.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

// #include <ppl.h>

const char *WINDOW_TITLE = "FEM BASIC 2D";

// GLUT state information
int g_viewportWidth = 800;
int g_viewportHeight = 600;

void display();
void reshape(GLint width, GLint height);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void KeyboardUpFunc(unsigned char key, int x, int y);
void SpecialFunc(int key, int x, int y);
void SpecialUpFunc(int key, int x, int y);
void Idle();
void DestroyWindow();
Eigen::Vector3d valueToColor( double d );

//using namespace boost;
using namespace c3ga;
using namespace std;
using namespace numerics;

class Camera
{
public:
	float		pos[3];
	float		fw[3];
	float		up[3];
	float		translateVel;
	float		rotateVel;

	Camera()
	{
		float		_pos[] = { 0, 0, 2};
		float		_fw[] = { 0, 0, -1 };
		float		_up[] = { 0, 1, 0 };

		translateVel = 0.005;
		rotateVel = 0.005;
		memcpy(pos, _pos, sizeof(float)*3);
		memcpy(fw, _fw, sizeof(float)*3);
		memcpy(up, _up, sizeof(float)*3);
	}

	void glLookAt()
	{
		gluLookAt( pos[0], pos[1], pos[2], fw[0],  fw[1],  fw[2], up[0],  up[1],  up[2] );
	}
};

class VertexBuffer
{
public:
	std::vector<Eigen::Vector3d> positions; //mesh vertex positions
	std::vector<Eigen::Vector3d> normals; //for rendering (lighting)
	std::vector<Eigen::Vector3d> colors; //for rendering (visual representation of values)
	int size;

	VertexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		positions.resize(size);
		normals.resize(size);
		colors.resize(size);
	}
	int get_size() { return size; }

};

class IndexBuffer {
public:
	std::vector<int> faces;
	int size;

	IndexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		faces.resize(size);
	}
	int get_size() { return size; }

};

Camera g_camera;
Mesh mesh;
vectorE3GA g_prevMousePos;
bool g_rotateModel = false;
bool g_rotateModelOutOfPlane = false;
rotor g_modelRotor = _rotor(1.0);
float g_dragDistance = -1.0f;
int g_dragObject;
bool g_showWires = true;


VertexBuffer vertexBuffer;
IndexBuffer indexBuffer;
std::shared_ptr<SparseMatrix> A;
Eigen::VectorXd right_hand_side;
Eigen::VectorXd solutionU;
Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
std::set<int> allconstraints;

/**
 * Add up all quantities associated with the element.
 * 
 * It computes the gradients on a reference 2D triagle then "push forward" it
 * to the triangle using the "differential" of the mapping. All of that follows
 * from the chain-rule as described in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleStiffnessElement(Vertex* v[3]) {

	Eigen::Matrix2d B, Binv;
	Eigen::Vector2d gradN[3];
	Eigen::Matrix3d elementMatrix;

	B(0,0) = v[1]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y();
	B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,1) = v[2]->p.y() - v[0]->p.y();

	Binv = B.inverse().transpose();

	double faceArea = 0.5 * abs(B.determinant());

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector2d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1));
	gradN[1] = Eigen::Vector2d(Binv(0,0), Binv(1,0));
	gradN[2] = Eigen::Vector2d(Binv(0,1), Binv(1,1));
	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
			elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]);
			if (i != j) {
				elementMatrix(j, i) = elementMatrix(i, j);
			}
		}
	}
	return elementMatrix;
}

/**
 * Extend computation of per-element stiffness matrix of triangle elements embedded in 3D space.
 * Original method only works for triangle elements on 2D space, see:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleStiffnessElementEmbedded(Vertex* v[3]) {

	Eigen::MatrixXd B(3, 2);
	Eigen::MatrixXd Binv(3, 2);
	Eigen::Vector3d gradN[3];
	Eigen::Matrix3d elementMatrix;

	B(0,0) = v[1]->p.x() - v[0]->p.x(); B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y(); B(1,1) = v[2]->p.y() - v[0]->p.y();
	B(2,0) = v[1]->p.z() - v[0]->p.z(); B(2,1) = v[2]->p.z() - v[0]->p.z();
    
	Binv = ((B.transpose() * B).inverse() * B.transpose()).transpose();

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector3d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1), -Binv(2,0) - Binv(2,1));
	gradN[1] = Eigen::Vector3d( Binv(0,0), Binv(1,0), Binv(2,0));
	gradN[2] = Eigen::Vector3d( Binv(0,1), Binv(1,1), Binv(2,1));
	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
			elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]);
			if (i != j) {
				elementMatrix(j, i) = elementMatrix(i, j);
			}
		}
	}
	return elementMatrix;
}

/**
 * Extend computation of per-element mass matrix of triangle elements embedded in 3D space.
 * Original method only works for triangle elements on 2D space, see:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleMassElement(Vertex* v[3], bool useMidpoints = true) {
	std::function<double(const Eigen::Vector2d &)> N[3];
	Eigen::Vector2d M_0, M_1, M_2;
	Eigen::Matrix3d massMatrix;

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();
 
	N[0] = [](const Eigen::Vector2d &U) { return 1 - U.x() - U.y(); };
	N[1] = [](const Eigen::Vector2d &U) { return U.x(); };
	N[2] = [](const Eigen::Vector2d &U) { return U.y(); };

	if (useMidpoints) {
		// we use triangle mid-points instead of vertices for exact quadrature rule
		M_0 = 0.5 * (Eigen::Vector2d(0,0) + Eigen::Vector2d(1,0)); 
		M_1 = 0.5 * (Eigen::Vector2d(1,0) + Eigen::Vector2d(0,1));
		M_2 = 0.5 * (Eigen::Vector2d(0,1) + Eigen::Vector2d(0,0));
	} else {
		M_0 = Eigen::Vector2d(0,0); 
		M_1 = Eigen::Vector2d(1,0);
		M_2 = Eigen::Vector2d(0,1);
	}

	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K / 3 [ N_i(M1) N_j(M1) + N_i(M2) N_j(M2) + N_i(M3) N_j(M3)]
			massMatrix(i, j) = (faceArea / 3.0) * (N[i](M_0) * N[j](M_0) + N[i](M_1) * N[j](M_1) + N[i](M_2) * N[j](M_2));
			if (i != j) {
				massMatrix(j, i) = massMatrix(i, j);
			}
		}
	}
	return massMatrix;
}

/**
 * Assemble the stiffness matrix. It does not take into account boundary conditions.
 * Boundary conditions will be applied when linear system is pre-factored (LU decomposition)
 * Original method can be found in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
std::shared_ptr<SparseMatrix> AssembleMatrix(Mesh *mesh, double delta_t) {
	std::shared_ptr<SparseMatrix> A(new SparseMatrix(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3d stiffnessMatrix, massMatrix;
	double wij;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		stiffnessMatrix = AssembleStiffnessElementEmbedded(v);
		//massMatrix = AssembleMassElement(v);
		for( int i = 0 ; i < 3 ; ++i ) {
			for (int j = 0 ; j < 3 ; ++j ) {
				if (i < j) continue; // since stifness matrix is symmetric
				//wij = massMatrix(i, j) + delta_t * stiffnessMatrix(i, j);
				wij = delta_t * stiffnessMatrix(i, j);
				(*A)(v[i]->ID, v[j]->ID) += wij;
				if (i != j) {
					(*A)(v[j]->ID, v[i]->ID) += wij;
				}
			}
		}
	}
	return A;
}

std::shared_ptr<SparseMatrix> AssembleDiagonalMassMatrix(Mesh *mesh) {
	std::shared_ptr<SparseMatrix> A(new SparseMatrix(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3d stiffnessMatrix, massMatrix;
	double wij;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		massMatrix = AssembleMassElement(v, false); // mass matrix is diagonal when useMidpoints = false
		wij = massMatrix(0, 0);
		(*A)(v[0]->ID, v[0]->ID) += wij;
		wij = massMatrix(1, 1);
		(*A)(v[1]->ID, v[1]->ID) += wij;
		wij = massMatrix(2, 2);
		(*A)(v[2]->ID, v[2]->ID) += wij;
	}
	return A;
}

void IplusMinvTimesA(std::shared_ptr<SparseMatrix> M, std::shared_ptr<SparseMatrix> A)
{
	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		SparseMatrix::RowIterator aIter = A->iterator(i);
		double oneOverVertexOneRingArea = 1.0 / (*M)(i, i);
		for (; !aIter.end(); ++aIter)
		{
			auto j = aIter.columnIndex();
			(*A)(i, j) *= oneOverVertexOneRingArea;
			if (i == j) {
				(*A)(i, j) += 1.0; // this completes the (I + M^-1 L)
			}
		}
	}
}

bool is_constrained(std::set<int>& constraints, int vertex)
{
	return constraints.find(vertex) != constraints.end();
}

/**
 * Apply boundaty conditions to stiffness matrix and pre-factor it 
 * using sparse LU decomposition.
 */
void PreFactor(std::shared_ptr<SparseMatrix> A, std::set<int>& constraints, Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>& solver)
{

	Eigen::SparseMatrix<double> Lc = Eigen::SparseMatrix<double>(A->numRows(), A->numColumns());

	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		if (!is_constrained(constraints, i))
		{
			SparseMatrix::RowIterator aIter = A->iterator(i);
			for (; !aIter.end(); ++aIter)
			{
				auto j = aIter.columnIndex();
				Lc.insert(i, j) = (*A)(i, j);
			}
		}
		else
		{
			Lc.insert(i, i) = 1.0;
		}
	}

	Lc.makeCompressed();
	solver.compute(Lc);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Error: " << "Prefactor failed." << std::endl;
		exit(1);
	}

}

int main(int argc, char* argv[])
{
	/**
	 * Load the FEM mesh
	 */
	mesh.readFEM("lake_nodes.txt", "lake_elements.txt");
	//mesh.readOBJ("cactus1.obj");
	mesh.CenterAndNormalize();
	mesh.computeNormals();

	// GLUT Window Initialization:
	glutInit (&argc, argv);
	glutInitWindowSize(g_viewportWidth, g_viewportHeight);
	glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow(WINDOW_TITLE);

	// Register callbacks:
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutKeyboardUpFunc(KeyboardUpFunc);
	glutSpecialFunc(SpecialFunc);
	glutSpecialUpFunc(SpecialUpFunc);
	glutIdleFunc(Idle);
	atexit(DestroyWindow);

	InitializeDrawing();

	vertexBuffer.resize(mesh.numVertices());
	indexBuffer.resize(mesh.numFaces() * 3);

	/**
	 * Initialize the vertex-buffer for OpenGL rendering purposes
	 */
	for( Vertex& vertex : mesh.getVertices())
	{
		vertexBuffer.positions[vertex.ID] = vertex.p;
		vertexBuffer.normals[vertex.ID] = vertex.n;
		vertexBuffer.colors[vertex.ID] = valueToColor(0);
	}
	
	double edgeCount = 0;
	double edgeLength = 0;
	/**
	 * Initialize the index-buffer for OpenGL rendering purposes
	 */
	for (Face& face : mesh.getFaces()) {
		int i = face.ID;
		int	v1 = face.edge->vertex->ID;
		int	v2 = face.edge->next->vertex->ID;
		int	v3 = face.edge->next->next->vertex->ID;
		indexBuffer.faces[i * 3 + 0] = v1;
		indexBuffer.faces[i * 3 + 1] = v2;
		indexBuffer.faces[i * 3 + 2] = v3;
		
		edgeLength += (face.edge->vertex->p - face.edge->next->vertex->p).norm();
		edgeLength += (face.edge->next->next->vertex->p - face.edge->vertex->p).norm();
		edgeLength += (face.edge->next->vertex->p - face.edge->next->next->vertex->p).norm();
		edgeCount += 3;
	}

	edgeLength /= edgeCount;

	/**
	 * Assemble the stiffness sparse matrix
	 */
	A = AssembleMatrix(&mesh, edgeLength);
	std::shared_ptr<SparseMatrix> M;
	M = AssembleDiagonalMassMatrix(&mesh);
	IplusMinvTimesA(M, A);

	/**
	 * Setup the right-hand-side of the linear system - including boundary conditions
	 */
	right_hand_side = Eigen::VectorXd(mesh.numVertices());
	right_hand_side.setZero(); // solve laplace's equation where RHS is zero
	
	for( Vertex& vertex : mesh.getVertices())
	{
		if (vertex.p.norm() < 2.5e-2) {
			right_hand_side(vertex.ID) = 1.0;
			allconstraints.insert(vertex.ID);
		}
	}

	/**
	 * Apply boundary conditions and perform sparse LU decomposition
	 */
	PreFactor(A, allconstraints, solver);

	/**
	 * Solve Laplace's equation
	 */
	solutionU = solver.solve(right_hand_side);

	/**
	 * Map solution values into colors
	 */
	double min_bc = *std::min_element(solutionU.begin(), solutionU.end());
	double max_bc = *std::max_element(solutionU.begin(), solutionU.end());

	for( Vertex& vertex : mesh.getVertices())
	{
		vertexBuffer.colors[vertex.ID] = valueToColor(solutionU[vertex.ID]);
	}

	glutMainLoop();

	return 0;
}

void display()
{
	/*
	 *	matrices
	 */
	glViewport( 0, 0, g_viewportWidth, g_viewportHeight );
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	pickLoadMatrix();
	GLpick::g_frustumFar = 1000.0;
	GLpick::g_frustumNear = .1;
	gluPerspective( 60.0, (double)g_viewportWidth/(double)g_viewportHeight, GLpick::g_frustumNear, GLpick::g_frustumFar );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);	//gouraud shading
	glClearDepth(1.0f);
	glClearColor( .75f, .75f, .75f, .0f );
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

	/*
	 *	estados
	 */
	glEnable(GL_CULL_FACE);		//face culling
	glCullFace( GL_BACK );
	glFrontFace( GL_CCW );
	glEnable(GL_DEPTH_TEST);	//z-buffer
	glDepthFunc(GL_LEQUAL);

	/*
	 *	iluminacion
	 */
	float		ambient[] = { .3f, .3f, .3f, 1.f };
	float		diffuse[] = { .3f, .3f, .3f, 1.f };
	float		position[] = { .0f, 0.f, 15.f, 1.f };
	float		specular[] = { 1.f, 1.f, 1.f };

	glLightfv( GL_LIGHT0, GL_AMBIENT, ambient );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0);
	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0125);
	glEnable(  GL_LIGHT0   );
	glEnable(  GL_LIGHTING );
	//glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, specular );
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.f );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glLoadIdentity();

	g_camera.glLookAt();

	glLightfv( GL_LIGHT0, /*GL_SPOT_DIRECTION*/GL_POSITION, position );

	glPushMatrix();

	rotorGLMult(g_modelRotor);

	if (GLpick::g_pickActive) glLoadName((GLuint)-1);

	double alpha = 1.0;

	//glEnable (GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//alpha = 0.5;

	//Mesh-Faces Rendering
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL /*GL_LINE GL_FILL GL_POINT*/);
	glEnable (GL_POLYGON_OFFSET_FILL);
	glPolygonOffset (1., 1.);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable( GL_COLOR_MATERIAL );
	if (GLpick::g_pickActive) glLoadName((GLuint)10);

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
	glNormalPointer(GL_DOUBLE, 0, &vertexBuffer.normals[0]);
	glColorPointer(3, GL_DOUBLE, 0, &vertexBuffer.colors[0]);

	// draw the model
	glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
	// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	if (g_showWires)
	{
		if (!GLpick::g_pickActive)
		{
			//Mesh-Edges Rendering (superimposed to faces)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE /*GL_LINE GL_FILL GL_POINT*/);
			glColor4d(.5, .5, .5, alpha);
			glDisable(GL_LIGHTING);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
			// draw the model
			glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
			// deactivate vertex arrays after drawing
			glDisableClientState(GL_VERTEX_ARRAY);
			glEnable(GL_LIGHTING);
		}
	}

	glDisable( GL_COLOR_MATERIAL );
	glDisable(GL_POLYGON_OFFSET_FILL);

	//glDisable (GL_BLEND);

	glPopMatrix();

	glutSwapBuffers();
}

Eigen::Vector3d valueToColor( double d )
{
	static Eigen::Vector3d	c0 = Eigen::Vector3d( 1, 1, 1);
	static Eigen::Vector3d	c1 = Eigen::Vector3d( 1, 1, 0);
	static Eigen::Vector3d	c2 = Eigen::Vector3d( 0, 1, 0);
	static Eigen::Vector3d	c3 = Eigen::Vector3d( 0, 1, 1);
	static Eigen::Vector3d	c4 = Eigen::Vector3d( 0, 0, 1);

	if( d < 0.25 )
	{
		double alpha = (d - 0.0) / (0.25-0.0);
		return (1.0 - alpha) * c0 + alpha * c1;
	}
	else if( d < 0.5 )
	{
		double alpha = (d - 0.25) / (0.5-0.25);
		return (1.0 - alpha) * c1 + alpha * c2;
	}
	else if( d < 0.75 )
	{
		double alpha = (d - 0.5) / (0.75-0.5);
		return (1.0 - alpha) * c2 + alpha * c3;
	}
	else
	{
		double alpha = (d - 0.75) / (1.0-0.75);
		return (1.0 - alpha) * c3 + alpha * c4;
	}
}


void reshape(GLint width, GLint height)
{
	g_viewportWidth = width;
	g_viewportHeight = height;

	// redraw viewport
	glutPostRedisplay();
}

vectorE3GA mousePosToVector(int x, int y) {
	x -= g_viewportWidth / 2;
	y -= g_viewportHeight / 2;
	return _vectorE3GA((float)-x * e1 - (float)y * e2);
}

void MouseButton(int button, int state, int x, int y)
{
	g_rotateModel = false;

	if (button == GLUT_LEFT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);

		if(g_dragObject == -1 || g_dragObject == 10 )
		{
			vectorE3GA mousePos = mousePosToVector(x, y);
			g_rotateModel = true;

			if ((_Float(norm_e(mousePos)) / _Float(norm_e(g_viewportWidth * e1 + g_viewportHeight * e2))) < 0.2)
				g_rotateModelOutOfPlane = true;
			else g_rotateModelOutOfPlane = false;
		}
	}

	if (button == GLUT_RIGHT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);
	}
}

void MouseMotion(int x, int y)
{
	if (g_rotateModel )
	{
		// get mouse position, motion
		vectorE3GA mousePos = mousePosToVector(x, y);
		vectorE3GA motion = mousePos - g_prevMousePos;

		if (g_rotateModel)
		{
			// update rotor
			if (g_rotateModelOutOfPlane)
				g_modelRotor = exp(g_camera.rotateVel * (motion ^ e3) ) * g_modelRotor;
			else 
				g_modelRotor = exp(0.00001f * (motion ^ mousePos) ) * g_modelRotor;
		}

		// remember mouse pos for next motion:
		g_prevMousePos = mousePos;

		// redraw viewport
		glutPostRedisplay();
	}
}

void SpecialFunc(int key, int x, int y)
{
	switch(key) {
		case GLUT_KEY_F1 :
			{
				int mod = glutGetModifiers();
				if(mod == GLUT_ACTIVE_CTRL || mod == GLUT_ACTIVE_SHIFT )
				{
				}
			}
			break;
		case GLUT_KEY_UP:
			{
			}
			break;
		case GLUT_KEY_DOWN:
			{
			}
			break;
	}
}

void SpecialUpFunc(int key, int x, int y)
{
}

void KeyboardUpFunc(unsigned char key, int x, int y)
{
	if(key == 'w' || key == 'W')
	{
		g_showWires = !g_showWires;
		glutPostRedisplay();
	}
}

void Idle()
{
	// redraw viewport
}

void DestroyWindow()
{
	ReleaseDrawing();
}

