#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <omp.h>
#include <fftw3.h>
#include <stdlib.h> 
#include <string>
#include <fstream>
using namespace std;
#define _USE_MATH_DEFINES
// global constants
const double G_scaled = 39.5 ; // Time in years, distances in AU, mass in solar masses
double G = 1;         // gravitational constant
double dt = 0.05;    // time interval
int   N = 2 ;           // particle number
const int gridN = 201;  // # of grid
const int gridNk = 5;          // # of grid in Z direction
const int total_grids = gridNk * gridN * gridN ;
const int T = 800; // simulation time steps
double Et , Es;    // energy for check 
double L = 100;      // box size
double Lz = L * gridNk / gridN;       // box size
const double PI = 3.1416;       
double dx = L / gridN;    // grid size 
fstream file; //宣告 file
int num_threads = 4; // for openMP

// 宣告所需矩陣
//vector<double> m(N, 0.0);  // mass
//vector<double> x(N, 0.0);  // x position
//vector<double> y(N, 0.0);  // y position
//vector<double> z(N, 0.0);  // z position
//vector<double> vx(N, 0.0);  // x velocity
//vector<double> vy(N, 0.0);  // y velocity
//vector<double> vz(N, 0.0);  // z velocity

void BC_for_particle(int p , double* x, double* y, double* z)
{
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (x[p] < -0.5 * L) { x[p] = fmod(x[p] + 0.5 * L, L) + 0.5 * L; }
        }
        #pragma omp section
        {
            if (x[p] > 0.5 * L) { x[p] = fmod(x[p] + 0.5 * L, L) - 0.5 * L; }
        }
        #pragma omp section
        {
            if (y[p] < -0.5 * L) { y[p] = fmod(y[p] + 0.5 * L, L) + 0.5 * L; }
        }
        #pragma omp section
        {
            if (y[p] > 0.5 * L) { y[p] = fmod(y[p] + 0.5 * L, L) - 0.5 * L; }
        }
        #pragma omp section
        {
            if (z[p] < -0.5 * Lz) { z[p] = fmod(z[p] + 0.5 * Lz, Lz) + 0.5 * Lz; }
        }
        #pragma omp section
        {
            if (z[p] > 0.5 * Lz) { z[p] = fmod(z[p] + 0.5 * Lz, Lz) - 0.5 * Lz; }
        }
    }
}
void output_phi(string name, double* maze)
{
    fstream phifile;
    phifile.open(name, ios::out);
    for (int i = 0; i < gridNk * gridN * gridN; i++) {
        printf("Output: %d / %d\n", i, total_grids);
        phifile << maze[i] << "\n";
    }
}
void NGP(int p, double i, double j, double k, double* rho, double* x, double* y, double* z , double* m)
{
    int aa, bb, cc;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (i >= gridN) { aa = i - gridN; }
            else if (i < 0) { aa = i + gridN; }
            else { aa = i; }
        }
        #pragma omp section
        {
            if (j >= gridN) { bb = j - gridN; }
            else if (j < 0) { bb = j + gridN; }
            else { bb = j; }
        }
        #pragma omp section
        {
            if (k >= gridNk) { cc = k - gridNk; }
            else if (k < 0) { cc = k + gridNk; }
            else { cc = k; }
        }
    }
        #pragma omp critical
    {
        rho[cc * gridN * gridN + bb * gridN + aa] += m[p] / (pow(dx, 3));
    }
}
void TSC(int p,double i, double j, double k, double *rho, double* x, double* y, double* z, double* m)
{
    double rx, ry, rz, wx, wy, wz;
    rx = ry = rz = wx = wy = wz = 0;
    for (int a = i - 1;a < i + 2;a++)
    for (int b = j - 1;b < j + 2;b++)
    for (int c = k - 1;c < k + 2;c++) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                rx = sqrt(pow((a * dx - (L / 2 - dx / 2)) - x[p], 2));
                if (rx <= dx / 2) { wx = 0.75 - pow(rx / dx, 2); }
                else if (rx <= 3 * dx / 2 && rx > dx / 2) { wx = pow(1.5 - rx / dx, 2) * 0.5; }
                else { wx = 0; }
            }
            #pragma omp section           
            {
                ry = sqrt(pow((b * dx - (L / 2 - dx / 2)) - y[p], 2));
                if (ry <= dx / 2) { wy = 0.75 - pow(ry / dx, 2); }
                else if (ry <= 3 * dx / 2 && ry > dx / 2) { wy = pow(1.5 - ry / dx, 2) * 0.5; }
                else { wy = 0; }
            }
            #pragma omp section
            {
                rz = sqrt(pow((c * dx - (Lz / 2 - dx / 2)) - z[p], 2));
                if (rz <= dx / 2) { wz = 0.75 - pow(rz / dx, 2); }
                else if (rz <= 3 * dx / 2 && rz > dx / 2) { wz = pow(1.5 - rz / dx, 2) * 0.5; }
                else { wz = 0; }
            }
                    
        }
        int aa, bb, cc;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (a >= gridN) { aa = a - gridN; }
                else if (a < 0) { aa = a + gridN; }
                else { aa = a; }
            }
            #pragma omp section
            {
                if (b >= gridN) { bb = b - gridN; }
                else if (b < 0) { bb = b + gridN; }
                else { bb = b; }
            }
            #pragma omp section
            {
                if (c >= gridNk) { cc = c - gridNk; }
                else if (c < 0) { cc = c + gridNk; }
                else { cc = c; }
            }
        }
        #pragma omp critical
        {
            rho[cc * gridN * gridN + bb * gridN + aa] += m[p] * wx * wy * wz / (pow(dx, 3));
        }
    }
}
void compute_rho(double* m,double* rho, double* x, double* y, double* z ) {
    // Zero-out rho for reuse
    memset(rho, 0, gridNk * gridN * gridN * sizeof(double));
    //#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int i, j, k;
        if (file.is_open()) {
            if (p != N - 1) { file << x[p] << "," << y[p] << "," << z[p] <<","; }
            else { file << x[p] << "," << y[p] << "," << z[p] << "\n"; }
        }
        i = floor((x[p] - (-L / 2)) / dx);
        j = floor((y[p] - (-L / 2)) / dx);
        k = floor((z[p] - (-Lz / 2)) / dx);
        //#pragma omp critical
        TSC( p,i, j, k, rho,x ,y, z, m);
        //NGP(p, i, j, k, rho, x, y, z, m);
        
    }
}
void compute_phi_k(fftw_complex* rho_k ) {
    double n_i, n_j, n_k, kx_i, ky_j, kz_k , scaling_factor;
    //#pragma omp for
    for (int k = 0; k < gridNk ; k++) {
        n_k = (k >= gridNk / 2) ? (gridNk-k) : k;
        kz_k = n_k * 2 * PI / Lz;
        for (int j = 0; j < gridN; j++) {
            n_j = (j >= gridN / 2) ? (gridN-j) : j;    // Negative frequency mapping in y
            ky_j = n_j * 2  * PI / L;          // Spatial frequency in y
            for (int i = 0; i < (gridN/2+1) ; i++) {
                n_i = i;
                if (n_i == 0 && n_j == 0 && n_k ==0) continue;
                kx_i = n_i * 2  * PI / L;      // Spatial frequency in x
                double fac = -1.0 * (pow(kx_i, 2) + pow(ky_j, 2) + pow(kz_k, 2))/(4 * PI* G);
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
                //scaling_factor = -4 * PI * G / k_sq;
                //rho_k[k + (gridN >> 1 + 1)*( j + gridN * i)][0] = 0;
                //rho_k[k  + gridN * j + gridN * gridN * i][0] *= (scaling_factor);
                //rho_k[k * gridN * (gridN >> 1 + 1) + j * (gridN >> 1 + 1) + i][1] *= (scaling_factor);
            }
        }
    }
}
void compute_accelerations(double* acc_x, double* acc_y, double* acc_z, double* phi) {
    // 2 delta_d for finite difference and N*N for FFTW normalization
    double scaling_factor = -2 * dx * gridNk * gridN * gridN;
    #pragma omp parallel
    {
        #pragma omp for collapse(3) nowait
        for (int k = 0; k < gridNk; k++) {
            for (int j = 0; j < gridN; j++) {
                for (int i = 1; i < gridN - 1; i++) {
                    acc_x[k * gridN * gridN + j * gridN + i] = (-phi[k * gridN * gridN + j * gridN + i - 1] + phi[k * gridN * gridN + j * gridN + i + 1]) / scaling_factor;
                }
            }
        }
        #pragma omp for collapse(3) nowait
        for (int k = 0; k < gridNk; k++) {
            for (int j = 1; j < gridN - 1; j++) {
                for (int i = 0; i < gridN; i++) {
                    acc_y[k * gridN * gridN + j * gridN + i] = (-phi[k * gridN * gridN + (j-1) * gridN + i ] + phi[k * gridN * gridN + (j+1) * gridN + i ]) / scaling_factor;
                }
            }
        }
        #pragma omp for collapse(3) nowait
        for (int k = 1; k < gridNk - 1; k++) {
            for (int j = 0; j < gridN; j++) {
                for (int i = 0; i < gridN; i++) {
                    acc_z[k * gridN * gridN + j * gridN + i] = (-phi[(k -1)* gridN * gridN + j * gridN + i ] + phi[(k+1) * gridN * gridN + j * gridN + i ]) / scaling_factor;
                }
            }
        }
        // Boundary conditions for phi are periodic
        // Up-down edges
        #pragma omp for nowait
        for (int i = 0; i < gridN; i++)
            for (int j = 0; j < gridN; j++) {
                acc_z[j * gridN + i] = (-phi[(gridNk - 1) * gridN * gridN + j * gridN + i] + phi[gridN * gridN + j * gridN + i]) / scaling_factor;
                acc_z[(gridNk - 1) * gridN * gridN + j * gridN + i] = (-phi[(gridNk - 2) * gridN * gridN + j * gridN + i] + phi[j * gridN + i]) / scaling_factor;
            }
        // Left-right edges
        #pragma omp for nowait
        for (int j = 0; j < gridN; j++)
            for (int k = 0; k < gridNk; k++) {
                acc_x[k * gridN * gridN + j * gridN] = (-phi[k * gridN * gridN + j * gridN + (gridN - 1)] + phi[k * gridN * gridN + j * gridN + 1]) / scaling_factor;
                acc_x[k * gridN * gridN + j * gridN + (gridN - 1)] = (-phi[k * gridN * gridN + j * gridN + gridN - 2] + phi[k * gridN * gridN + j * gridN]) / scaling_factor;
            }
        // Top-bottom edges
        #pragma omp for
        for (int i = 0; i < gridN; i++)
            for (int k = 0; k < gridNk; k++) {
                acc_y[k * gridN * gridN + i] = (-phi[k * gridN * gridN + (gridN - 1) * gridN + i] + phi[k * gridN * gridN + gridN + i]) / scaling_factor;
                acc_y[k * gridN * gridN + (gridN - 1) * gridN + i] = (-phi[k * gridN * gridN + (gridN - 2) * gridN + i] + phi[k * gridN * gridN + i]) / scaling_factor;
            }
    }
}
void update_particle_KDK( double* x, double* y, double* z, double* vx, double* vy, double* vz, double* a_x, double* a_y, double* a_z)
{
    for (int p = 0; p < N; p++) {
        int i, j, k;
        #pragma omp parallel
        {
            #pragma omp single
            {
                i = floor((x[p] - (-L / 2)) / dx);
                j = floor((y[p] - (-L / 2)) / dx);
                k = floor((z[p] - (-Lz / 2)) / dx);
            }
            #pragma omp sections
            {
                #pragma omp section
                {
                    vx[p] = vx[p] + a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    x[p] = x[p] + vx[p] * dt;
                    vx[p]  = vx[p] + a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vy[p] = vy[p] + a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    y[p] = y[p] + vy[p] * dt;
                    vy[p] = vy[p] + a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vz[p] = vz[p] + a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    z[p] = z[p] + vz[p] * dt;
                    vz[p] = vz[p] + a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
            }
            #pragma omp barrier
            BC_for_particle(p,x,y,z);
        }
    }
}
void update_particle_DKD( double* x, double* y, double* z, double* vx, double* vy, double* vz, double* a_x, double* a_y, double* a_z)
{
    for (int p = 0; p < N; p++) {
        int i, j, k;
        #pragma omp parallel
        {
            #pragma omp single
            {
                i = floor((x[p] - (-L / 2)) / dx);
                j = floor((y[p] - (-L / 2)) / dx);
                k = floor((z[p] - (-Lz / 2)) / dx);
            }
            #pragma omp sections
            {
                #pragma omp section
                {
                    vx[p] = vx[p] + a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    x[p] = x[p] + vx[p] * dt;
                    vx[p] = vx[p] + a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vy[p] = vy[p] + a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    y[p] = y[p] + vy[p] * dt;
                    vy[p] = vy[p] + a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vz[p] = vz[p] + a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    z[p] = z[p] + vz[p] * dt;
                    vz[p] = vz[p] + a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
            }
            #pragma omp barrier
            BC_for_particle(p, x, y, z);
        }
    }
}
void calculate_energy(double* m , double* x , double* y , double* z) {
    double rx, ry, rz ,rr;
    rx = fabs(x[1] - x[0]);
    ry = fabs(y[1] - y[0]);
    rz = fabs(z[1] - z[0]);
    rr = sqrt(rx * rx + ry * ry + rz * rz);
    Es = -G * m[0] * m[1] / rr;
}

int main(int argc, char* argv[])
{
    fftw_complex* rho_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((gridN >> 1) + 1) * gridN * gridNk);
    double* rho = (double*)fftw_malloc(sizeof(double) * gridNk * gridN * gridN);
    double* phi = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* a_x = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* a_y = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* a_z = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* m = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double* z = (double*)malloc(N * sizeof(double));
    double* vx = (double*)malloc(N * sizeof(double));
    double* vy = (double*)malloc(N * sizeof(double));
    double* vz = (double*)malloc(N * sizeof(double));

    m[0] = 400.0 ;
    m[1] = 400.0 ;
    x[0] = -4.0;
    x[1] = 4.0;
    y[0] = 0.0;
    y[1] = -0.0;
    z[0] = 0.0;
    z[1] = 0.0;
    vx[0] = 0.0;
    vx[1] = -0.0;
    vy[0] = 5.0;
    vy[1] = -5.0;
    vz[0] = 0.0;
    vz[1] = -0.0; 
    Et = -G * m[0] * m[1] / fabs(x[1] - x[0]);
    omp_set_num_threads(num_threads);
    fftw_plan_with_nthreads(num_threads);
    fftw_plan rho_plan = fftw_plan_dft_r2c_3d(gridNk, gridN, gridN, rho, rho_k, FFTW_MEASURE);
    fftw_plan phi_plan = fftw_plan_dft_c2r_3d(gridNk, gridN, gridN, rho_k, phi, FFTW_MEASURE);
    //vector< vector< vector<double> > > maze  ( gridN, vector< vector<double> > ( gridN, vector<double>(gridN, 0.0) ) ); // density
    file.open("Data.csv", ios::out);
    double t_start = omp_get_wtime();
    
    for (int t = 1; t < T; t++) {
        printf("Progress: %d / %d\n", t, T);
        //printf("compute_rho\n");
        compute_rho( m, rho, x, y, z);
        //printf("fftw_execute\n");
        #pragma omp single
        fftw_execute(rho_plan);
        //printf("compute_phi_k\n");
        compute_phi_k(rho_k);
        #pragma omp single
        fftw_execute(phi_plan);
        //printf("compute_accelerations\n");
        compute_accelerations(a_x, a_y, a_z, phi);
        //printf("update_particle\n");
        //for (int p = 0; p < N; p++) { printf("XYZ:%f %f %f\n", x[p], y[p], z[p]); }
        update_particle_KDK(x, y, z, vx, vy, vz, a_x, a_y, a_z);
        //update_particle_DKD(x, y, z, vx, vy, vz, a_x, a_y, a_z);
        //for (int p = 0; p < N; p++) { printf("XYZ:%f %f %f\n", x[p], y[p], z[p]); }
        //calculate_energy(m,x,y,z);
        //double error = 100*(Es - Et) / Et;
        //printf("Error; %f\n", error);
        //output_phi("HITHIT.txt");
    }
    double t_end = omp_get_wtime();
    double total_time_ms = t_end - t_start;
    printf("Time: %f\n", total_time_ms);  
    //output_phi("HITHIT.txt",phi);
    fftw_destroy_plan(rho_plan);
    fftw_destroy_plan(phi_plan);
    printf("free memory!\n");/*
    free(rho);
    //free(phi);
    printf("free a memory!\n");
    free(a_x);
    free(a_y);
    free(a_z);
    printf("free m memory!\n");
    free(m);
    printf("free x memory!\n");
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    fftw_free(rho_k);*/
    return(0);
}
