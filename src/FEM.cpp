/**
 * BASIC 2D FEM on a 3D MESH
 *
 * This follows the method described in: 
 * a) Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 * b) Hans Peter Langtangen, "Introduction to finite element methods", 2014 
 *    http://hplgit.github.io/INF5620/doc/pub/H14/fem/html/main_fem.html
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

