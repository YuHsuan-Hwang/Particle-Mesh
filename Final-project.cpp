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
double dt = 0.1;    // time interval
int   N = 2 ;           // particle number
const int gridN = 513;  // # of grid
const int gridNk = 9;          // # of grid in Z direction
const int total_grids = gridNk * gridN * gridN ;
const int T = 250;
double Et , Es;
double L = 120;      // box size
double Lz = L * gridNk / gridN;       // box size
const double PI = 3.1416;
double dx = L / gridN;    // grid size 
float mass = 0.0; // for checking the total mass
//fftw_complex* rho_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * gridN * gridN * gridNk);
fftw_complex* rho_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((gridN >> 1) + 1) * gridN * gridNk);
double* rho = (double*)fftw_malloc(sizeof(double) * gridNk * gridN * gridN);
double* phi = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
double* a_x = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
double* a_y = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
double* a_z = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
fstream file;
vector<double> m(N, 0.0);  // mass
vector<double> x(N, 0.0);  // x position
vector<double> y(N, 0.0);  // y position
vector<double> z(N, 0.0);  // z position
vector<double> vx(N, 0.0);  // x velocity
vector<double> vy(N, 0.0);  // y velocity
vector<double> vz(N, 0.0);  // z velocity


void output_phi(string name)
{
    fstream phifile;
    phifile.open(name, ios::out);
    int total = gridNk * gridN * gridN;
    /*
    if (!file)
    {
        cerr << "Can't open file!\n";

        exit(1);     //在不正常情形下，中斷程式的執行
    }
    */
    for (int i = 0; i < gridNk * gridN * gridN; i++) {
        printf("Output: %d / %d\n", i, total_grids);
        phifile << phi[i] << "\n";
    }
}
void compute_rho(int N, int gridN , double* rho) {
    // Zero-out rho for reuse
    memset(rho, 0, gridNk * gridN * gridN * sizeof(double));
    float rx, ry, rz; //the distance between the center of the grid and the particle
    float wx, wy, wz;  //operator for density distribution
    rx = ry = rz = 0;
    wx = wy = wz = 0;
    //#pragma omp parallel for
    for (int p = 0; p < N; p++) {
        int i, j, k;
        if (file.is_open()) {
            if (p != N - 1) { file << x[p] << "," << y[p] << "," << z[p] <<","; }
            else { file << x[p] << "," << y[p] << "," << z[p] << "\n"; }
        }
            i = (x[p] - (-L / 2)) / dx;
            j = (y[p] - (-L / 2)) / dx;
            k = (z[p] - (-Lz / 2)) / dx;
        //#pragma omp critical
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
                            if (a < 0) { aa = a + gridN; }
                            if (a >= 0 && a < gridN) { aa = a; }
                        }
                        #pragma omp section
                        {
                            if (b >= gridN) { bb = b - gridN; }
                            if (b < 0) { bb = b + gridN; }
                            if (b >= 0 && b < gridN) { bb = b; }
                        }
                        #pragma omp section
                        {
                            if (c >= gridNk) { cc = c - gridNk; }
                            if (c < 0) { cc = c + gridNk; }
                            if (c >= 0 && c < gridNk) { cc = c; }
                        }
                        //{ cc = (c >= gridNk) ? (c - gridNk) : c; cc = (c < 0) ? (c + gridNk) : c; }
                    }
                    #pragma omp critical
                    {
                        //printf("PABC: %d %d %d\n", aa, bb, cc);
                        rho[cc * gridN * gridN + bb * gridN + aa] += m[p] * wx * wy * wz / (pow(dx, 3));
                    }
                }
        }
}
void compute_phi_k(int gridN, double L, fftw_complex* rho_k, double G) {
    double n_i, n_j, n_k, kx_i, ky_j, kz_k , k_sq, scaling_factor;
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
                double fac = -1.0 * (pow(kx_i, 2) + pow(ky_j, 2) + pow(kz_k, 2))/(4 * PI * G);
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
}
void compute_accelerations(int gridN, double* acc_x, double* acc_y, double* acc_z, double* phi) {
    // 2 delta_d for finite difference and N*N for FFTW normalization
    double scaling_factor = 2 * dx * gridNk * gridN * gridN;
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
        /*
        printf("Ax: %f\n", a_x[180085]);
        printf("PH: %f\n", phi[180070]);
        printf("PH: %f\n", phi[180080]);
        printf("Ax: %f\n", a_x[180115]);
        printf("PH: %f\n", phi[180120]);
        printf("PH: %f\n", phi[180125]);*/
    }
}
void update_particle_KDK( double* a_x, double* a_y, double* a_z)
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
                    vx[p] = a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    x[p] += vx[p] * dt;
                    vx[p] = a_x[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vy[p] = a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    y[p] += vy[p] * dt;
                    vy[p] = a_y[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
                #pragma omp section
                {
                    vz[p] = a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                    z[p] += vz[p] * dt;
                    vz[p] = a_z[k * gridN * gridN + j * gridN + i] * 0.5 * dt;
                }
            }
            #pragma omp barrier
            #pragma omp sections
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
    }
}
void update_particle_DKD(double* a_x, double* a_y, double* a_z)
{
    for (int p = 0; p < N; p++) {
        int i, j, k;
        i = floor((x[p] - (-L / 2)) / dx);
        j = floor((y[p] - (-L / 2)) / dx);
        k = floor((z[p] - (-Lz / 2)) / dx);
        x[p] += vx[p] * 0.5 * dt;
        y[p] += vy[p] * 0.5 * dt;
        z[p] += vz[p] * 0.5 * dt;
        vx[p] = a_x[k * gridN * gridN + j * gridN + i] * dt;
        vy[p] = a_y[k * gridN * gridN + j * gridN + i] * dt;
        vz[p] = a_z[k * gridN * gridN + j * gridN + i] * dt;
        x[p] += vx[p] * 0.5 * dt;
        y[p] += vy[p] * 0.5 * dt;
        z[p] += vz[p] * 0.5 * dt;
        if (x[p] < -0.5 * L) x[p] = fmod(x[p] + 0.5 * L, L) + L;
        if (x[p] > 0.5 * L) x[p] = fmod(x[p] + 0.5 * L, L);
        if (y[p] < -0.5 * L) y[p] = fmod(y[p] + 0.5 * L, L) + L;
        if (y[p] > 0.5 * L) y[p] = fmod(y[p] + 0.5 * L, L);
        if (z[p] < -0.5 * L) z[p] = fmod(z[p] + 0.5 * L, L) + L;
        if (z[p] > 0.5 * L) z[p] = fmod(z[p] + 0.5 * L, L);
    }
}
void calculate_energy() {
    double rx, ry, rz ,rr;
    rx = fabs(x[1] - x[0]);
    ry = fabs(y[1] - y[0]);
    rz = fabs(z[1] - z[0]);
    rr = sqrt(rx * rx + ry * ry + rz * rz);
    Es = -G * m[0] * m[1] / rr;
}



int main(int argc, char* argv[])
{
    //int argc, char *argv[]
    m = { 100.0 ,100.0 , 20.0 , 90.0 , 30.0 , 6.5};
    x = { -25.0, 25.0, -20.0 , -0.0, 4.2 };
    y = { 0.0, 0.0, -0.0 , 25.0 ,0.0};
    z = { 0.0, 0.0, 0.0 , 0.0 ,0.0};
    vy = { 2.0, -2.0, 0.0 , 0.0 ,0.0 };
    Et = -G * m[0] * m[1] / fabs(x[1] - x[0]);
    //printf("total E: %f\n", Et);
    int num_threads = 4;
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
        compute_rho(N, gridN, rho);
        //printf("fftw_execute\n");
        #pragma omp single
        fftw_execute(rho_plan);
        //printf("compute_phi_k\n");
        compute_phi_k(gridN, L, rho_k, G);
        #pragma omp single
        fftw_execute(phi_plan);
        //printf("compute_accelerations\n");
        compute_accelerations(gridN, a_x, a_y, a_z, phi);
        //printf("update_particle\n");
        //for (int p = 0; p < N; p++) { printf("XYZ:%f %f %f\n", x[p], y[p], z[p]); }
        update_particle_KDK(a_x, a_y, a_z);
        //for (int p = 0; p < N; p++) { printf("XYZ:%f %f %f\n", x[p], y[p], z[p]); }
        calculate_energy();
        double error = 100*(Es - Et) / Et;
        //memset(phi, 0, gridNk * gridN * gridN * sizeof(double));
        //intf("Error; %f\n", error);
        //output_phi("HITHIT.txt");
    }
    double t_end = omp_get_wtime();
    double total_time_ms = t_end - t_start;
    printf("Time: %f\n", total_time_ms);  
    //output_phi("HITHIT.txt");
    //printf("dfwrpkgoekir");
    //fftw_destroy_plan(rho_plan);
    //fftw_destroy_plan(phi_plan);
    //free(rho);
    //free(a_x);
    //free(a_y);
    //free(a_z);
    //fftw_free(rho_k);
    return(0);

}
