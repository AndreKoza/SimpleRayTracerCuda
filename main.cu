#include <iostream>
#include <fstream>
#include <limits>
#include <chrono>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float flt_max = std::numeric_limits<float>::max();

#ifndef FLT_MAX
#define FLT_MAX flt_max
#endif

#include "material.h"
#include "camera.h"
#include "sphere.h"
#include "hitable_list.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) 
{
	if (result) 
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state)
{
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++)
	{
		hit_record rec;
		// use 0.001 instead of 0 to avoid shadow acne (in this case leads to exception (don't know why))
		if ((*world)->hit(cur_ray, 0.001, FLT_MAX, rec))
		{
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else
			{
				return vec3(0.0f, 0.0f, 0.0f);
			}
		}
		else
		{
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			// Linear interpolate between white and some other color
			vec3 c = (1.0f - t)*vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * c;
		}
	}
	return vec3(0.f, 0.f, 0.f); // exceeded "recursion"
}


__global__ void rand_init(curandState *rand_state) 
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(1984, 0, 0, rand_state);
	}
}


__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
	{
		return;
	}
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
	{
		return;
	}
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}


#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
						new lambertian(vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				float choose_mat = RND;
				vec3 center(a + RND, 0.2f, b + RND);
				if (choose_mat < 0.8f)
				{
					d_list[i++] = new sphere(center, 0.2f,
						new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
				}
				else if (choose_mat < 0.95f)
				{
					d_list[i++] = new sphere(center, 0.2f,
						new metal(vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND));
				}
				else
				{
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0f, new dielectric(1.5f));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0f, new lambertian(vec3(0.4, 0.2, 0.1f)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0f, new metal(vec3(0.7, 0.6, 0.5), 0.0f));
		*rand_state = local_rand_state;
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0f; // (lookfrom - lookat).length();
		float aperture = 0.1f;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20.0f,
			float(nx) / float(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera)
{
	for (int i = 0; i < 22 * 22 + 1 + 3; i++)
	{
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}


int main()
{
    auto start = std::chrono::system_clock::now();
    std::ofstream output;
    output.open("picture.ppm");

	int nx = 1200;
	int ny = 800;
	int ns = 100;
	int tx = 8;
	int ty = 8;
    output << "P3\n" << nx << " " << ny << "\n255\n";

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// make our world of hitables & the camera
	hitable **d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Output FB as Image
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());
			output << ir << " " << ig << " " << ib << "\n";
		}
	}


	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();

    output.close();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Total time: " << diff.count() << " s\n";
    std::cin.ignore();
    return 0;
}
    