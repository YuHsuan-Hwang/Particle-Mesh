#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <omp.h>
#include <fftw3.h>
#include <stdlib.h> 
#include <string>
#include <fstream>
#include <time.h>
using namespace std;
#define _USE_MATH_DEFINES
// global constants
const double PI = 3.1416;
const double G = 1;         // gravitational constant
const int N = 800000 ;           // particle number
const int gridN = 64;           // # of grid in XY direction
const int gridNk = gridN;          // # of grid in Z direction
const int total_grids = gridNk * gridN * gridN ;
const double L = 30;        // box size in XY direction
const double Lz = L * gridNk / gridN;       // box size in Z direction
const double dx = L / gridN;    // grid size
const double dt = 0.1;    // time interval
const int T = 100; // simulation time steps
const int sample_rate = 1;     //資料輸出頻率
double Et, Es;         // energy for check error
int num_threads = 4;     // for openMP 
//給粒子的邊界條件，當粒子撞到牆壁會穿過，從另一變回來
void BC_for_particle(int p , double* x, double* y, double* z)
{
    if (x[p] < -0.5 * L) { x[p] = fmod(x[p] + 0.5 * L, L) + 0.5 * L; }
    if (x[p] > 0.5 * L) { x[p] = fmod(x[p] + 0.5 * L, L) - 0.5 * L; }
    if (y[p] < -0.5 * L) { y[p] = fmod(y[p] + 0.5 * L, L) + 0.5 * L; }
    if (y[p] > 0.5 * L) { y[p] = fmod(y[p] + 0.5 * L, L) - 0.5 * L; }
    if (z[p] < -0.5 * Lz) { z[p] = fmod(z[p] + 0.5 * Lz, Lz) + 0.5 * Lz; }
    if (z[p] > 0.5 * Lz) { z[p] = fmod(z[p] + 0.5 * Lz, Lz) - 0.5 * Lz; }
}
//可輸出三圍矩陣，一般是拿來輸出密度場或位能場
void output_maze(string name, double* maze)
{
    fstream phifile;
    phifile.open(name, ios::out);
    for (int i = 0; i < gridNk * gridN * gridN; i++) {
        //printf("Output: %d / %d\n", i, total_grids);
        phifile << maze[i] << "\n";
    }
}
//計算密度場演算法其一
void NGP(int p, double i, double j, double k, double* rho, double* x, double* y, double* z , double* m)
{
    int aa, bb, cc;
    if (i >= gridN) { aa = i - gridN; }
    else if (i < 0) { aa = i + gridN; }
    else { aa = i; }
    if (j >= gridN) { bb = j - gridN; }
    else if (j < 0) { bb = j + gridN; }
    else { bb = j; }
    if (k >= gridNk) { cc = k - gridNk; }
    else if (k < 0) { cc = k + gridNk; }
    else { cc = k; }
    rho[cc * gridN * gridN + bb * gridN + aa] += m[p] / (pow(dx, 3));
}
//計算密度場演算法其二
void CIC(int n, double* m, double* rho, double* x, double* y, double* z )
{
    vector< vector<int> > x_index(N, vector<int>(2, 0));
    vector< vector<int> > y_index(N, vector<int>(2, 0));
    vector< vector<int> > z_index(N, vector<int>(2, 0));
    vector<double> wx(N, 0.0);
    vector<double> wy(N, 0.0);
    vector<double> wz(N, 0.0);
    x_index[n][0] = int((x[n] + L / 2.0) / dx);
    wx[n] = (x[n] + L / 2.0) / dx - x_index[n][0];
    if (wx[n] > 0.5) {
        x_index[n][1] = x_index[n][0];
        x_index[n][0] += 1;
    }
    else {
        x_index[n][1] = x_index[n][0] + 1;
        wx[n] = 1.0 - wx[n];
    }

    // printf("%d %d %f\n",x_index[n][0],x_index[n][1],wx[n]);

    y_index[n][0] = int((y[n] + L / 2.0) / dx);
    wy[n] = (y[n] + L / 2.0) / dx - y_index[n][0];
    if (wy[n] > 0.5) {
        y_index[n][1] = y_index[n][0];
        y_index[n][0] += 1;
    }
    else {
        y_index[n][1] = y_index[n][0] + 1;
        wy[n] = 1.0 - wy[n];
    }

    z_index[n][0] = int((z[n] + L / 2.0) / dx);
    wz[n] = (z[n] + L / 2.0) / dx - z_index[n][0];
    if (wz[n] > 0.5) {
        z_index[n][1] = z_index[n][0];
        z_index[n][0] += 1;
    }
    else {
        z_index[n][1] = z_index[n][0] + 1;
        wz[n] = 1.0 - wz[n];
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                rho[x_index[n][i] + gridN * y_index[n][j] + gridN * gridN * z_index[n][k]] += m[n] * abs(i - wx[n]) * abs(j - wy[n]) * abs(k - wz[n]) / dx / dx / dx;
            }
        }
    }
}
//計算密度場演算法其三
void TSC(int p,int i,int j, int k, double *rho, double* x, double* y, double* z, double* m , double* matrix)
{
    double rx, ry, rz, Wx, Wy, Wz;
    rx = ry = rz = Wx = Wy = Wz = 0;
    for (int c = k - 1;c < k + 2;c++)
    for (int b = j - 1;b < j + 2;b++)
    for (int a = i - 1;a < i + 2;a++) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                rx = sqrt(pow((a * dx - (L / 2 - dx / 2)) - x[p], 2));
                if (rx <= dx / 2) { Wx = 0.75 - pow(rx / dx, 2); }
                else if (rx <= 3 * dx / 2 && rx > dx / 2) { Wx = pow(1.5 - rx / dx, 2) * 0.5; }
                else { Wx = 0; }
            }
            #pragma omp section           
            {
                ry = sqrt(pow((b * dx - (L / 2 - dx / 2)) - y[p], 2));
                if (ry <= dx / 2) { Wy = 0.75 - pow(ry / dx, 2); }
                else if (ry <= 3 * dx / 2 && ry > dx / 2) { Wy = pow(1.5 - ry / dx, 2) * 0.5; }
                else { Wy = 0; }
            }
            #pragma omp section
            {
                rz = sqrt(pow((c * dx - (Lz / 2 - dx / 2)) - z[p], 2));
                if (rz <= dx / 2) { Wz = 0.75 - pow(rz / dx, 2); }
                else if (rz <= 3 * dx / 2 && rz > dx / 2) { Wz = pow(1.5 - rz / dx, 2) * 0.5; }
                else { Wz = 0; }
            }
                    
        }
        matrix[(a - (i - 1)) + (b - (j - 1)) * 3 + (c - (k - 1)) * 3 * 3 + p * 27] = Wx * Wy * Wz;
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
            rho[cc * gridN * gridN + bb * gridN + aa] += m[p] * Wx * Wy * Wz / (pow(dx, 3));
        }
    }
}
//計算密度場，需套用前面演算法函式
void compute_rho(double* m,double* rho, double* x, double* y, double* z, int t, double* matrix, int* index, double* data) {
    memset(rho, 0, gridNk * gridN * gridN * sizeof(double)); //清空原密度場
    #pragma omp parallel for
    for (int p = 0; p < N; p++) {
        //找尋 particle 位在空間矩陣的哪一個位置
        index[3 * p + 0] = floor((x[p] - (-L / 2)) / dx);
        index[3 * p + 1] = floor((y[p] - (-L / 2)) / dx);
        index[3 * p + 2] = floor((z[p] - (-Lz / 2)) / dx);
        //儲存該粒子所在的精準位置，供輸出檔案用
        data[p * 3 + 0 + t * N * 3] = x[p];
        data[p * 3 + 1 + t * N * 3] = y[p];
        data[p * 3 + 2 + t * N * 3] = z[p];
        //套用演算法
        //CIC(p, m, rho, x, y, z);
        //TSC( p,index[3 * p + 0], index[3 * p + 1], index[3 * p + 2], rho,x ,y, z, m, matrix);
        NGP(p, index[3 * p + 0], index[3 * p + 1], index[3 * p + 2], rho, x, y, z, m);
    }
}
//從rho(k)做 Poisson 轉成 phi(k)
void compute_phi_k(fftw_complex* rho_k ) {
    double n_i, n_j, n_k, kx_i, ky_j, kz_k;
    //#pragma omp for
    for (int k = 0; k < gridNk ; k++) {
        n_k = (k >= gridNk / 2) ? ( gridNk - k ) : k;
        kz_k = n_k * 2 * PI / Lz;
        for (int j = 0; j < gridN; j++) {
            n_j = (j >= gridN / 2) ? ( gridN - j) : j;   
            ky_j = n_j * 2  * PI / L;  
            for (int i = 0; i < (gridN/2+1) ; i++) {
                n_i = i;
                if (n_i == 0 && n_j == 0 && n_k ==0) continue;
                kx_i = n_i * 2  * PI / L;      // Spatial frequency in x
                double fac = -(4 * PI * G) /(pow(kx_i, 2) + pow(ky_j, 2) + pow(kz_k, 2));
                rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][0] *= fac;
                rho_k[k * gridN * (gridN / 2 + 1) + j * (gridN / 2 + 1) + i][1] *= fac;
            }
        }
    }
}
//從位能場算出加速度場
void compute_accelerations(double* acc_x, double* acc_y, double* acc_z, double* phi) {
    double scaling_factor = -2 * dx * gridN * gridN * gridNk; //for FFTW normalization
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
                    acc_y[k * gridN * gridN + j * gridN + i] = (-phi[k * gridN * gridN + (j - 1) * gridN + i] + phi[k * gridN * gridN + (j + 1) * gridN + i]) / scaling_factor;
                }
            }
        }
        #pragma omp for collapse(3) nowait
        for (int k = 1; k < gridNk - 1; k++) {
            for (int j = 0; j < gridN; j++) {
                for (int i = 0; i < gridN; i++) {
                    acc_z[k * gridN * gridN + j * gridN + i] = (-phi[(k - 1) * gridN * gridN + j * gridN + i] + phi[(k + 1) * gridN * gridN + j * gridN + i]) / scaling_factor;
                }
            }
        }
        // 設定邊界條件，假設空間是週期性
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
//加速度加權 for CIC
void CIC_compute_acc(double* x, double* y, double* z, double* u, double* acc_x, double* acc_y, double* acc_z)
{
    vector< vector<int> > x_index(N, vector<int>(2, 0));
    vector< vector<int> > y_index(N, vector<int>(2, 0));
    vector< vector<int> > z_index(N, vector<int>(2, 0));

    vector<double> wx(N, 0.0);
    vector<double> wy(N, 0.0);
    vector<double> wz(N, 0.0);
    for (int n = 0; n < N; n++) {

        int i, j, k;
        i = floor((x[n] - (-L / 2)) / dx);
        j = floor((y[n] - (-L / 2)) / dx);
        k = floor((z[n] - (-Lz / 2)) / dx);

        x_index[n][0] = int((x[n] + L / 2.0) / dx);
        wx[n] = (x[n] + L / 2.0) / dx - x_index[n][0];
        if (wx[n] > 0.5) {
            x_index[n][1] = x_index[n][0];
            x_index[n][0] += 1;
        }
        else {
            x_index[n][1] = x_index[n][0] + 1;
            wx[n] = 1.0 - wx[n];
        }
        // printf("%d %d %f\n",x_index[n][0],x_index[n][1],wx[n]);

        y_index[n][0] = int((y[n] + L / 2.0) / dx);
        wy[n] = (y[n] + L / 2.0) / dx - y_index[n][0];
        if (wy[n] > 0.5) {
            y_index[n][1] = y_index[n][0];
            y_index[n][0] += 1;
        }
        else {
            y_index[n][1] = y_index[n][0] + 1;
            wy[n] = 1.0 - wy[n];
        }
        z_index[n][0] = int((z[n] + L / 2.0) / dx);
        wz[n] = (z[n] + L / 2.0) / dx - z_index[n][0];
        if (wz[n] > 0.5) {
            z_index[n][1] = z_index[n][0];
            z_index[n][0] += 1;
        }
        else {
            z_index[n][1] = z_index[n][0] + 1;
            wz[n] = 1.0 - wz[n];
        }
        double ax_p = 0.0;
        double ay_p = 0.0;
        double az_p = 0.0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double ax_p_tmp = u[x_index[n][i] + 1 + gridN * y_index[n][j] + gridN * gridN * z_index[n][k]] - u[x_index[n][i] - 1 + gridN * y_index[n][j] + gridN * gridN * z_index[n][k]];
                    ax_p_tmp /= 2.0 * dx * total_grids;
                    ax_p -= ax_p_tmp * abs(i - wx[n]) * abs(j - wy[n]) * abs(k - wz[n]);
                    double ay_p_tmp = u[x_index[n][i] + gridN * (y_index[n][j] + 1) + gridN * gridN * z_index[n][k]] - u[x_index[n][i] + gridN * (y_index[n][j] - 1) + gridN * gridN * z_index[n][k]];
                    ay_p_tmp /= 2.0 * dx * total_grids;
                    ay_p -= ay_p_tmp * abs(i - wx[n]) * abs(j - wy[n]) * abs(k - wz[n]);
                    double az_p_tmp = u[x_index[n][i] + gridN * y_index[n][j] + gridN * gridN * (z_index[n][k] + 1)] - u[x_index[n][i] + gridN * y_index[n][j] + gridN * gridN * (z_index[n][k] - 1)];
                    az_p_tmp /= 2.0 * dx * total_grids;
                    az_p -= az_p_tmp * abs(i - wx[n]) * abs(j - wy[n]) * abs(k - wz[n]);
                }
            }
        }
        acc_x[k * gridN * gridN + j * gridN + i] = ax_p;
        acc_y[k * gridN * gridN + j * gridN + i] = ay_p;
        acc_z[k * gridN * gridN + j * gridN + i] = az_p;
    }
}
//加速度加權 for TSC
void TSC_compute_acc(int p, int i, int j, int k, double* matrix, double* acc_x, double* acc_y, double* acc_z)
{
    double ax, ay, az;
    ax = ay = az = 0;
    for (int a = i - 1; a < i + 2; a++)
        for (int b = j - 1; b < j + 2; b++)
            for (int c = k - 1; c < k + 2; c++) {
                //printf("ABC: %d %d %d\n", a, b, c);
                //printf("W: %f\n", matrix[p * 27 + (c - (k - 1)) * 3 * 3 + (b - (j - 1)) * 3 + (a - (i - 1))]);
                ax = ax + acc_x[c * gridN * gridN + b * gridN + a] * matrix[p * 27 + (c - (k - 1)) * 3 * 3 + (b - (j - 1)) * 3 + (a - (i - 1))];
                ay = ay + acc_y[c * gridN * gridN + b * gridN + a] * matrix[p * 27 + (c - (k - 1)) * 3 * 3 + (b - (j - 1)) * 3 + (a - (i - 1))];
                az = az + acc_z[c * gridN * gridN + b * gridN + a] * matrix[p * 27 + (c - (k - 1)) * 3 * 3 + (b - (j - 1)) * 3 + (a - (i - 1))];
            }
    acc_x[k * gridN * gridN + j * gridN + i] = ax;
    acc_y[k * gridN * gridN + j * gridN + i] = ay;
    acc_z[k * gridN * gridN + j * gridN + i] = az;
}
//KDK，若變換密度場演算法則須改內容
void update_particle_KDK(double* x, double* y, double* z, double* vx, double* vy, double* vz, double* a_x, double* a_y, double* a_z, double* matrix, double*phi, int* index)
{
    //CIC_compute_acc(x, y, z, phi, a_x, a_y, a_z);
    #pragma omp parallel for
    for (int p = 0; p < N; p++) {
        index[p * 3 + 0] = floor((x[p] - (-L / 2)) / dx);
        index[p * 3 + 1] = floor((y[p] - (-L / 2)) / dx);
        index[p * 3 + 2] = floor((z[p] - (-Lz / 2)) / dx);       
        //TSC_compute_acc(p, index[p * 3 + 0], index[p * 3 + 1], index[p * 3 + 2], matrix, a_x, a_y, a_z);
        vx[p] = vx[p] + a_x[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        x[p] = x[p] + vx[p] * dt;
        vx[p] = vx[p] + a_x[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        vy[p] = vy[p] + a_y[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        y[p] = y[p] + vy[p] * dt;
        vy[p] = vy[p] + a_y[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        vz[p] = vz[p] + a_z[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        z[p] = z[p] + vz[p] * dt;
        vz[p] = vz[p] + a_z[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * 0.5 * dt;
        BC_for_particle(p, x, y, z);
    }
}      
//DKD，若變換密度場演算法則須改內容
void update_particle_DKD(double* x, double* y, double* z, double* vx, double* vy, double* vz, double* a_x, double* a_y, double* a_z, double* matrix, int* index)
{
    //CIC_compute_acc(x, y, z, phi, a_x, a_y, a_z);
    #pragma omp parallel for
    for (int p = 0; p < N; p++) {
        index[p * 3 + 0] = floor((x[p] - (-L / 2)) / dx);
        index[p * 3 + 1] = floor((y[p] - (-L / 2)) / dx);
        index[p * 3 + 2] = floor((z[p] - (-Lz / 2)) / dx);
        //TSC_compute_acc(p, index[p * 3 + 0], index[p * 3 + 1], index[p * 3 + 2], matrix, a_x, a_y, a_z);
        x[p] = x[p] + vx[p] * 0.5 * dt;
        vx[p] = vx[p] + a_x[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] *  dt;
        x[p] = x[p] + vx[p] * 0.5 * dt;
        y[p] = y[p] + vy[p] * 0.5 * dt;
        vy[p] = vy[p] + a_y[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * dt;
        y[p] = y[p] + vy[p] * 0.5 * dt;
        z[p] = z[p] + vz[p] * 0.5 * dt;
        vz[p] = vz[p] + a_z[index[p * 3 + 2] * gridN * gridN + index[p * 3 + 1] * gridN + index[p * 3 + 0]] * dt;
        z[p] = z[p] + vz[p] * 0.5 * dt;
        BC_for_particle(p, x, y, z);
    }
}
//for checking error
void calculate_energy(double* m , double* x , double* y , double* z, double* vx, double* vy, double* vz) {
    double rx, ry, rz ,rr;
    rx = fabs(x[1] - x[0]);
    ry = fabs(y[1] - y[0]);
    rz = fabs(z[1] - z[0]);
    rr = sqrt(rx * rx + ry * ry + rz * rz);
    Es = 0.5 * m[0] * (vx[0] * vx[0] + vy[0] * vy[0] + vz[0] * vz[0]) + 0.5 * m[1] * (vx[1] * vx[1] + vy[1] * vy[1] + vz[1] * vz[1]) - G * m[0] * m[1] / rr;
}
//for N-body，可給予 N 個粒子隨機位置及速度或質量
void assign(int N, double* m, double* x, double* y, double* z, double* vx, double* vy, double* vz)
{
    srand(time(NULL));
    double min = - 0.3 * L ;
    double max = 0.3 * L ;
    for (int i = 0; i < N; i++) {
        m[i] = 2.0;
        x[i] = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        y[i] = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        z[i] = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        vx[i] = 0.0;
        vy[i] = 0.0;
        vz[i] = 0.0;
    }
}



int main(int argc, char* argv[])
{
    fftw_complex* rho_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((gridN >> 1) + 1) * gridN * gridNk);
    double* rho = (double*)fftw_malloc(sizeof(double) * gridNk * gridN * gridN);
    double* phi = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* m = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double* z = (double*)malloc(N * sizeof(double));
    double* vx = (double*)malloc(N * sizeof(double));
    double* vy = (double*)malloc(N * sizeof(double));
    double* vz = (double*)malloc(N * sizeof(double));
    double* a_x = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* a_y = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* a_z = (double*)malloc(gridNk * gridN * gridN * sizeof(double));
    double* data = (double*)malloc( T * 3 * N * sizeof(double)); //存粒子精準位置用，最後輸出檔案給派森畫圖
    int* index = (int*)malloc( 3 * N * sizeof(int));     //存粒子在空間矩陣上的位置用
    double* TSC_matrix = (double*)malloc( 27 * N * sizeof(double)); //TSC加權矩陣
    fstream file;             //宣告 file，準備給 data輸出位置資料用
    /*for (int i = 0; i < N; i++) {TSC_matrix[13 + i * 27] = 1.0;}*/
    
    assign(N, m, x, y, z, vx,vy,vz);  //分配 N個粒子隨機位置質量速度，僅可用於 NGP，因為其他的沒設邊界條件
    /*m[0] = 400.0;
    m[1] = 400.0;
    x[0] = 4.0;
    x[1] = -4.0;
    y[0] = 0.0;
    y[1] = -0.0;
    z[0] = 0.0;
    z[1] = -0.0;
    vx[0] = 0.0;
    vx[1] = -0.0;
    vy[0] = 5.0;
    vy[1] = -5.0;
    vz[0] = 0.0;
    vz[1] = -0.0;*/
    //Et = 0.5 * m[0] * (vx[0] * vx[0] + vy[0] * vy[0] + vz[0] * vz[0]) + 0.5 * m[1] * (vx[1] * vx[1] + vy[1] * vy[1] + vz[1] * vz[1]) -G * m[0] * m[1] / fabs(x[1] - x[0]);
    omp_set_num_threads(num_threads);       // for openMP parallelize
    fftw_plan_with_nthreads(num_threads);   // for openMP parallelize in fft
    fftw_plan rho_plan = fftw_plan_dft_r2c_3d(gridNk, gridN, gridN, rho, rho_k, FFTW_MEASURE);
    fftw_plan phi_plan = fftw_plan_dft_c2r_3d(gridNk, gridN, gridN, rho_k, phi, FFTW_MEASURE);
    //file.open("Data.csv", ios::out);
    double t_start = omp_get_wtime();   //紀錄迴圈開始時間
    for (int t = 0; t < T; t++) {
        printf("Progress: %d / %d\n", t, T);

        compute_rho( m, rho, x, y, z, t, TSC_matrix, index,data);

        #pragma omp single
        fftw_execute(rho_plan);

        compute_phi_k(rho_k);
        #pragma omp single
        fftw_execute(phi_plan);

        compute_accelerations(a_x, a_y, a_z, phi);

        update_particle_KDK(x, y, z, vx, vy, vz, a_x, a_y, a_z, TSC_matrix,phi, index);
        //update_particle_DKD(x, y, z, vx, vy, vz, a_x, a_y, a_z, TSC_matrix, index);
        
        //calculate_energy(m,x,y,z, vx, vy, vz);
        //double error = 100*(Es - Et) / Et;
        //printf("Error; %f\n", error); 
    }
    double t_end = omp_get_wtime();  //紀錄迴圈結束時間
    double total_time_ms = t_end - t_start; //跑所有迴圈總共時間
    printf("Time: %f\n", total_time_ms);  //輸出到底花了多少時間
    printf("Output data..,\n");
    for (int i = 0; i < N * 3 * T; i++) {
        if (file.is_open()) {
            if (fmod(i, N * 3) != N * 3-1) { file << data[i] << ","; }
            else { file << data[i] << "\n"; }
        }
    }  
    //output_maze("PHI.txt",phi);
    //fftw_destroy_plan(rho_plan);
    //fftw_destroy_plan(phi_plan);
    /*free(rho);
    //free(phi);
    free(a_x);
    free(a_y);
    free(a_z);
    free(m);
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    fftw_free(rho_k);*/
    return(0);
}
