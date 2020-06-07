#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>
//#include "NGP.h"
#include "CIC.h"
//#include "TSC.h"

using namespace std;

#define _USE_MATH_DEFINES


// global variables

//// constants

float G  = 1.0;    // gravitational constant
float dt = 1.0e-2; // time interval

//// set particles 

int N; // particle number
vector<double> m ( N, 0.0 );  // mass
vector<double> x ( N, 0.0 );  // x position
vector<double> y ( N, 0.0 );  // y position
vector<double> z ( N, 0.0 );  // z position
vector<double> vx( N, 0.0 );  // x velocity
vector<double> vy( N, 0.0 );  // y velocity
vector<double> vz( N, 0.0 );  // z velocity

//// set grids

int gridN = 8;           // grid number
double L  = 7;           // total length of the cube
double dx = L/(gridN-1); // length of one grid
// for gridN = 8, L = 7, The value of the grids would be: -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5  

vector< vector< vector<double> > > p  ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); // density
vector< vector< vector<double> > > u  ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); // potential
vector< vector< vector<double> > > ax ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); // acceleration
vector< vector< vector<double> > > ay ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); 
vector< vector< vector<double> > > az ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); 


// functions

double CalEnergy();
void   Drift( float );

// main function

int main( int argc, char *argv[] )
{

	// initial condition

	////two body motion

	N = 2;	
	m  = { 2.0, 1.0 };
	x  = {-0.5, 1.0 };
	y  = { 0.0, 0.0 };
	z  = { 0.0, 0.0 };
	vx = { 0.0, 0.0 };
	vy = { -1.0*sqrt(G/1.5/3.0), 2.0*sqrt(G/1.5/3.0) };
	vz = { 0.0, 0.0 }; 
	

	// calculate total energy
	
	double E0 = CalEnergy();
	printf("initial total energy: %f\n",E0);	


	// time parameters

	double t = 0.0;
	double period = 2.0*M_PI*sqrt( (x[1]*x[1]+y[1]*y[1])/(vx[1]*vx[1]+vy[1]*vy[1]) );
	double end_time = 1.0*period;
	printf("period: %f\n",period);


	// update particles

	while( t<end_time ){


		// particle mesh


		
		//// method1: DKD

		Drift( 0.5*dt );

		//Kick_NGP( dt );
		Kick_CIC( dt );
		//Kick_TSC( dt );

		Drift( 0.5*dt );




		/*
		//// method2: KDK

		Kick_NGP( 0.5*dt );
                //Kick_CIC( 0.5*dt );
                //Kick_TSC( 0.5*dt );

		Drift( dt );

                Kick_NGP( 0.5*dt );
                //Kick_CIC( 0.5*dt );
                //Kick_TSC( 0.5*dt );
		*/
	


		t += dt;
	
	}


	// calculate energy error
	
	double E = CalEnergy();
	double energy_err = abs( (E-E0)/E0 );
	printf("energy error: %f\n",energy_err);

}


double CalEnergy()
{
	double E = 0.0;
	for( int n = 0; n < N; n++ ){
		
		// kinetic energy
		E += 0.5*m[n]*( vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n] );
		
		// gravitational energy
		for( int i = 0; i < N; i++ ){
			if ( i!=n ){
				double dx = x[i]-x[n];
				double dy = y[i]-y[n];
				double dz = z[i]-z[n];
				double dist = sqrt( dx*dx + dy*dy + dz*dz );
				E += -0.5*G*m[i]*m[n]/dist;
			}
		}
	}

	return E;

}


void Drift( float delta_t )
{
	for( int n = 0; n < N; n++ ){
		x[n] += vx[n] * delta_t;
		y[n] += vy[n] * delta_t;
		z[n] += vz[n] * delta_t;
	}
}
