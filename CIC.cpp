#include <cstdio> 
#include <cstdlib>
#include <math.h>
#include <vector>

using namespace std;

extern float G;
extern vector<double> m, x, y, z, vx, vy, vz;
extern int N, gridN;
extern double L, dx;
extern vector< vector< vector<double> > > p, u, ax, ay, az;


void Kick_CIC( float delta_t )
{

	// Mass Deposit

        vector< vector<int> > x_index ( N, vector<int>(2,0) );
        vector< vector<int> > y_index ( N, vector<int>(2,0) );
        vector< vector<int> > z_index ( N, vector<int>(2,0) );

        vector<double> wx ( N, 0.0 );
        vector<double> wy ( N, 0.0 );
        vector<double> wz ( N, 0.0 );

        for( int n = 0; n < N; n++ ){

                x_index[n][0] = int( ( x[n]+L/2.0 )/dx );
                wx[n] = ( x[n]+L/2.0 )/dx - x_index[n][0];
                if (wx[n]>0.5){
                        x_index[n][1] = x_index[n][0];
                        x_index[n][0] += 1;
                }
                else{
                        x_index[n][1] = x_index[n][0] + 1;
                        wx[n] = dx - wx[n];
                }

                printf("%d %d %f\n",x_index[n][0],x_index[n][1],wx[n]);
        
		y_index[n][0] = int( ( y[n]+L/2.0 )/dx );
                wy[n] = ( y[n]+L/2.0 )/dx - y_index[n][0];
                if (wy[n]>0.5){
                        y_index[n][1] = y_index[n][0];
                        y_index[n][0] += 1;
                }
                else{
                        y_index[n][1] = y_index[n][0] + 1;
                        wy[n] = dx - wy[n];
                }

		z_index[n][0] = int( ( z[n]+L/2.0 )/dx );
                wz[n] = ( z[n]+L/2.0 )/dx - z_index[n][0];
                if (wz[n]>0.5){
                        z_index[n][1] = z_index[n][0];
                        z_index[n][0] += 1;
                }
                else{
                        z_index[n][1] = z_index[n][0] + 1;
                        wz[n] = dx - wz[n];
                }

		for( int i = 0; i < 2; i++ ){
			for( int j = 0; j < 2; j++ ){
				for( int k = 0; k < 2; k++ ){
					p[ x_index[n][i] ][ y_index[n][j] ][ z_index[n][k] ] += m[n]*abs(i-wx[n])*abs(j-wy[n])*abs(k-wz[n]) /dx/dx/dx ;
				}
			}
		}
	
	}

	// check total mass
	double m_tmp = 0.0;
	for( int i = 0; i < gridN; i++ ){
		for( int j = 0; j < gridN; j++ ){
			for( int k = 0; k < gridN; k++ ){
				m_tmp += p[i][j][k];
			}
		}
	}
	m_tmp *= dx*dx*dx ;
	printf("ckeck total mass: %f\n",m_tmp);







}

