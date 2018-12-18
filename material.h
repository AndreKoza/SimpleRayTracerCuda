#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"


// Returns a random point within unit sphere. First a point within the unit cube is calculated and then rejected if outside
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        // Range of random numbers: [0, 1). Start is center of sphere: function 2*x-1 reaches values between [-1, 1).
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

// Standard reflection equation (incident angle = reflected angle etc.)
__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v,n) * n;
}

// Standard refraction equation, a bit hard to derive and strange variable names but for now I stick to tutorial names
__device__ bool refract(const vec3& v, const vec3& n, float ni_over_ti, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_ti*ni_over_ti*(1.0f-dt*dt);
    if (discriminant > 0)
    {
        refracted = ni_over_ti*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
    {
        return false;
    }
}

// Reflectivity varies with angle -> this is an approximation for that (haven't looked into that)
__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f-cosine),5.0f);
}

class material
{
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};


class lambertian : public material
{
    public:
		__device__ lambertian(const vec3& a) :
            albedo(a)
        {
        }

		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const
        {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            scattered = ray(rec.p, target-rec.p);
            attenuation = albedo;
            return true;
        }

        vec3 albedo;
};


class metal : public material
{
    public:
        __device__ metal(const vec3& a, float f = 0.0f) :
            albedo(a)
        {
            if (f < 1.0f)
            {
                fuzz = f;
            }
            else
            {
                fuzz = 1;
            }
        }

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const
        {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            // See definition of dot product: If dot product > 0 -> angle is sharp (spitzer Winkel)
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }

        vec3 albedo;
        float fuzz;
};


class dielectric : public material
{
    public:
        __device__ dielectric(float ri) :
            ref_idx(ri)
        {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const
        {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 1.0);
            vec3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f)
            {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
				//cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
				//cosine = sqrt(1.0f - ref_idx * ref_idx*(1 - cosine * cosine));
                cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }
            else
            {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }
            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            {
                reflect_prob = schlick(cosine, ref_idx);
            }
            else
            {
                reflect_prob = 1.0f;
            }
            if (curand_uniform(local_rand_state) < reflect_prob)
            {
                scattered = ray(rec.p, reflected);
            }
            else
            {
                scattered = ray(rec.p, refracted);
            }
            return true;
        }

        float ref_idx;
};

#endif