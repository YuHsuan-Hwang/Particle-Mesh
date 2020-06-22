#include <cstdio> 
#include <cstdlib>
#include <math.h>
#include <vector>
#include "/usr/local/include/fftw3.h"

using namespace std;
#define _USE_MATH_DEFINES

extern float G;
extern vector<double> m, x, y, z, vx, vy, vz;
extern int N, gridN;
extern double L, dx;
extern vector< vector< vector<double> > > u;

vector<double> Linspace( float, float, int );
fftw_complex* compute_phi_k( int, double, fftw_complex*, double );

void Kick_CIC( double delta_t )
{

	// Mass Deposit
	vector< vector< vector<double> > > p  ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) );
	vector< vector< vector<double> > > u  ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) );
        
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
                        wx[n] = 1.0 - wx[n];
                }

                // printf("%d %d %f\n",x_index[n][0],x_index[n][1],wx[n]);
        
		y_index[n][0] = int( ( y[n]+L/2.0 )/dx );
                wy[n] = ( y[n]+L/2.0 )/dx - y_index[n][0];
                if (wy[n]>0.5){
                        y_index[n][1] = y_index[n][0];
                        y_index[n][0] += 1;
                }
                else{
                        y_index[n][1] = y_index[n][0] + 1;
                        wy[n] = 1.0 - wy[n];
                }

		z_index[n][0] = int( ( z[n]+L/2.0 )/dx );
                wz[n] = ( z[n]+L/2.0 )/dx - z_index[n][0];
                if (wz[n]>0.5){
                        z_index[n][1] = z_index[n][0];
                        z_index[n][0] += 1;
                }
                else{
                        z_index[n][1] = z_index[n][0] + 1;
                        wz[n] = 1.0 - wz[n];
                }

		for( int i = 0; i < 2; i++ ){
			for( int j = 0; j < 2; j++ ){
				for( int k = 0; k < 2; k++ ){
					p[ x_index[n][i] ][ y_index[n][j] ][ z_index[n][k] ] += m[n]*abs(i-wx[n])*abs(j-wy[n])*abs(k-wz[n]) /dx/dx/dx ;
				}
			}
		}
	
	}

	/*	
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
	*/

	/*
	// print p
	for( int i = 0; i < gridN; i++ ){
		for( int j = 0; j < gridN; j++ ){
			for( int k = 0; k < gridN; k++ ){
				printf("%f ",p[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}
	*/

	// Solve Poisson by FFT
	
        fftw_complex *in, *out;
	fftw_plan plan_foward, plan_backward;
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *gridN*gridN*gridN );
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *gridN*gridN*gridN );
	
	for( int l = 0; l < gridN*gridN*gridN; l++ ){
		
		int k = l /gridN/gridN;
		int j = ( l%(gridN*gridN) ) /gridN;
		int i = l - (gridN*gridN)*k - gridN*j;

		in[l][0] = p[i][j][k];
		in[l][1] = 0.0;

	}
	
	/*
	// print in
	for( int l = 0; l < gridN*gridN*gridN; l++ ){
		printf("%f ",in[l][0]);
	}
	printf("\n");
	*/
		
	plan_foward = fftw_plan_dft_3d( gridN, gridN, gridN, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_backward = fftw_plan_dft_3d( gridN, gridN, gridN, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute( plan_foward );

        for( int l = 0; l < gridN*gridN*gridN; l++ ){
                for( int i = 0; i < 2; i++ ){
                        out[l][i] /= gridN*gridN*gridN;
                }
        }
	//out = compute_phi_k(gridN, L, out, G);
	
	vector<double> k_vec = Linspace( 0, L, gridN );
	// vector<double> k_vec = {0.   ,  0.125,  0.25 ,  0.375, -0.5  , -0.375, -0.25 , -0.125};
	for( int l = 0; l < gridN*gridN*gridN; l++ ){
		
		int k = l /gridN/gridN;
		int j = ( l%(gridN*gridN) ) /gridN;
		int i = l - (gridN*gridN)*k - gridN*j;
		if ( (i!=0)&(j!=0)&(k!=0) ){

			for( int i = 0; i < 2; i++ ){
				out[l][i] /= 2.0*M_PI*2.0*M_PI*( -(k_vec[i]*k_vec[i])-(k_vec[j]*k_vec[j])-(k_vec[k]*k_vec[k]) );
				//out[l][i] *= 4.0*M_PI*G;
			}
		}
	}
	
	fftw_execute( plan_backward );

        fftw_destroy_plan( plan_foward );
	fftw_destroy_plan( plan_backward );

	for( int l = 0; l < gridN*gridN*gridN; l++ ){

		int k = l /gridN/gridN;
		int j = ( l%(gridN*gridN) ) /gridN;
		int i = l - (gridN*gridN)*k - gridN*j;

		u[i][j][k] = in[l][0];
	}

	fftw_free(in);
	fftw_free(out);
	
	/*
	// check result of FFT	
	for( int k = 0; k < gridN; k++ ){
                for( int j = 0; j < gridN; j++ ){
                        for( int i = 0; i < gridN; i++ ){
				
				printf("%f ",u[i][j][k]);
				//if ( abs(u[i][j][k]-p[i][j][k]) > 0.001 ){
				//	printf("wrong %f %f\n", p[i][j][k], u[i][j][k]);
				//}

			}
			printf("\n");
		}
		printf("\n");
	}	
	*/

	// calculate a and update v of each particle

	for( int n = 0; n < N; n++ ){
		double ax_p = 0.0;
		double ay_p = 0.0;
		double az_p = 0.0;
		for( int i = 0; i < 2; i++ ){
                        for( int j = 0; j < 2; j++ ){
                                for( int k = 0; k < 2; k++ ){
					
					double ax_p_tmp = u[ x_index[n][i]+1 ][ y_index[n][j] ][ z_index[n][k] ] - u[ x_index[n][i]-1 ][ y_index[n][j] ][ z_index[n][k] ];
					ax_p_tmp /= 2.0*dx;
					ax_p -= ax_p_tmp*abs(i-wx[n])*abs(j-wy[n])*abs(k-wz[n]);
					
					double ay_p_tmp = u[ x_index[n][i] ][ y_index[n][j]+1 ][ z_index[n][k] ] - u[ x_index[n][i] ][ y_index[n][j]-1 ][ z_index[n][k] ];
                                        ay_p_tmp /= 2.0*dx;
                                        ay_p -= ay_p_tmp*abs(i-wx[n])*abs(j-wy[n])*abs(k-wz[n]);

					double az_p_tmp = u[ x_index[n][i] ][ y_index[n][j] ][ z_index[n][k]+1 ] - u[ x_index[n][i] ][ y_index[n][j] ][ z_index[n][k]-1 ];
                                        az_p_tmp /= 2.0*dx;
                                        az_p -= az_p_tmp*abs(i-wx[n])*abs(j-wy[n])*abs(k-wz[n]);

				}
			}
		}
		printf("ax_p %f \n",ax_p);
		vx[n] += ax_p * delta_t;
		vy[n] += ay_p * delta_t;
		vz[n] += az_p * delta_t;
	}




}



vector<double> Linspace( float start, float end, int number ) {

        vector<double> vec(number);
        for( int i = 0; i < vec.size(); i++ ) {
                vec[i] = i*(end-start)/(number-1.0);
                //printf("%f ",vec[i]);
        }

        return vec;
}

fftw_complex* compute_phi_k(int gridN, double L, fftw_complex* rho_k, double G) {
    int gridNk = gridN;
    double n_i, n_j, n_k, kx_i, ky_j, kz_k , k_sq, scaling_factor;
    //#pragma omp for
    for (int k = 0; k < gridNk ; k++) {
        n_k = (k >= gridNk / 2) ? (gridNk-k) : k;
        //n_k = k;
        kz_k = n_k * 2 * M_PI / L;
        for (int j = 0; j < gridN; j++) {
            n_j = (j >= gridN / 2) ? (gridN-j) : j;    // Negative frequency mapping in y
            ky_j = n_j * 2  * M_PI / L;          // Spatial frequency in y
            for (int i = 0; i < (gridN/2+1) ; i++) {
                n_i = i;
                //n_i = (i >= gridN / 2) ? (gridN - i) : i;
                if (n_i == 0 && n_j == 0 && n_k ==0) continue;
                kx_i = n_i * 2  * M_PI / L;      // Spatial frequency in x
                //k_sq = kx_i * kx_i + ky_j * ky_j + kz_k * kz_k;
                double fac = -1.0 * (pow(kx_i, 2) + pow(ky_j, 2) + pow(kz_k, 2))/(4 * M_PI * G);
                if (fabs(fac) < 1e-14)
                {
                    rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][0] = 0;
                    rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][1] = 0;
                }
                else
                {
                    rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][0] /= fac;
                    rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][1] /= fac;
                }
                //scaling_factor = 1;
                //scaling_factor = -4 * PI * G / k_sq;
                //scaling_factor = -1 / k_sq;
                //rho_k[k + (gridN >> 1 + 1)*( j + gridN * i)][0] = 0;
                //rho_k[k  + gridN * j + gridN * gridN * i][0] *= (scaling_factor);
                //rho_k[k + gridN * j + gridN * gridN * i][1] *= (scaling_factor);
                //rho_k[k * gridN * (gridN /2 + 1) + j * (gridN /2 + 1) + i][0] *= (scaling_factor);
                //rho_k[k * gridN * (gridN >> 1 + 1) + j * (gridN >> 1 + 1) + i][1] *= (scaling_factor);
            }
        }
    }
    return rho_k;
}
