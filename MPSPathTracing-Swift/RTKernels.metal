//
//  RTKernels.metal
//  MPSPathTracing-Swift
//
//  Created by Jaewon Choi on 2023/03/03.
//

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include "Utils.h"

using namespace metal;

struct Ray {
    packed_float3 origin;
    uint mask;
    packed_float3 direction;
    float maxDistance;
    float3 color;
};

struct Intersection {
    float distance;
    int primitiveIndex;
    float2 coordinates;
};

kernel void rayKernel(uint2 tid [[thread_position_in_grid]],
                      constant RTUniforms& uniforms,
                      device Ray* rays,
                      texture2d<unsigned int> randomTex,
                      texture2d<float, access::write> dstTex) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) {
        return;
    }

    unsigned int rayIdx = tid.y * uniforms.width + tid.x;

    device Ray& ray = rays[rayIdx];

    float2 pixel = (float2)tid;

    unsigned int offset = randomTex.read(tid).x;

    float2 r = float2(halton(offset + uniforms.frameIndex, 0),
                      halton(offset + uniforms.frameIndex, 1));

    pixel += r;

    float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;

    constant Camera& camera = uniforms.camera;

    ray.origin = camera.position;
    ray.direction = normalize(uv.x * camera.right
                              + uv.y * camera.up
                              + camera.forward);
    ray.mask = RAY_MASK_PRIMARY;
    ray.maxDistance = INFINITY;
    ray.color = float3(1.0f, 1.0f, 1.0f);

    dstTex.write(float4(0.0f, 0.0f, 0.0f, 0.0f), tid);
}

template <typename T>
inline T interpolateVertexAttribute(device T* attributes, Intersection intersection) {
    float3 uvw;
    uvw.xy = intersection.coordinates;
    uvw.z = 1.0f - uvw.x - uvw.y;

    unsigned int triangleIndex = intersection.primitiveIndex;

    T T0 = attributes[triangleIndex * 3 + 0];
    T T1 = attributes[triangleIndex * 3 + 1];
    T T2 = attributes[triangleIndex * 3 + 2];

    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;

    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);

    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

inline void sampleAreaLight(constant AreaLight& light,
                                   float2 u,
                                   float3 position,
                                   thread float3& lightDirection,
                                   thread float3& lightColor,
                                   thread float& lightDistance) {
    u = u * 2.0f - 1.0f;

    float3 samplePosition = light.position +
                            light.right * u.x +
                            light.up * u.y;

    // Normalizing lightDirection
    lightDirection = samplePosition - position;

    lightDistance = length(lightDirection);
    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
    lightDirection *= inverseLightDistance;

    lightColor = light.color;
    lightColor *= (inverseLightDistance * inverseLightDistance); // Light falls off with the inverse square of the distance to the intersection point.
    lightColor *= saturate(dot(-lightDirection, light.forward)); // Light falls off with the cosine of angle between the intersection point and the light source
}

inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    float3 up = normal;
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f))); // Find a arbitrary direction perpendicular to the normal. This will become the "right" vector.
    float3 forward = cross(right, up);

    return sample.x * right + sample.y * up + sample.z * forward;
}

kernel void shadeKernel(uint2 tid [[thread_position_in_grid]],
                        constant RTUniforms& uniforms,
                        device Ray* rays,
                        device Ray* shadowRays,
                        device Intersection* intersections,
                        device float3* vertexColors,
                        device float3* vertexNormals,
                        device uint* triangleMasks,
                        constant unsigned int& bounce,
                        texture2d<unsigned int> randomTex,
                        texture2d<float, access::write> dstTex) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) {
        return;
    }

    unsigned int rayIdx = tid.y * uniforms.width + tid.x;
    device Ray& ray = rays[rayIdx];
    device Ray& shadowRay = shadowRays[rayIdx];
    device Intersection& intersection = intersections[rayIdx];

    ray.maxDistance = -1.0f;
    shadowRay.maxDistance = -1.0f;

    if (ray.maxDistance < 0.0f || intersection.distance < 0.0f) {
        return;
    }

    uint mask = triangleMasks[intersection.primitiveIndex];

    if (mask != TRIANGLE_MASK_GEOMETRY) {
        dstTex.write(float4(uniforms.light.color, 1.0f), tid);
        return;
    }

    float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;
    float3 surfaceNormal = interpolateVertexAttribute(vertexNormals, intersection);
    surfaceNormal = normalize(surfaceNormal);

    unsigned int offset = randomTex.read(tid).x;

    float2 r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 4 + 0),
                      halton(offset + uniforms.frameIndex, 2 + bounce * 4 + 1));

    float3 lightDirection;
    float3 lightColor;
    float lightDistance;
    float3 color = ray.color;

    sampleAreaLight(uniforms.light, r, intersectionPoint, lightDirection, lightColor, lightDistance);

    lightColor *= saturate(dot(surfaceNormal, lightDirection));
    color *= interpolateVertexAttribute(vertexColors, intersection);

    shadowRay.origin = intersectionPoint + surfaceNormal * 1e-3f;
    shadowRay.direction = lightDirection;
    shadowRay.mask = RAY_MASK_SHADOW;
    shadowRay.maxDistance = lightDistance - 1e-3f;
    shadowRay.color = lightColor * color;

    r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 4 + 2),
               halton(offset + uniforms.frameIndex, 2 + bounce * 4 + 3));

    float3 sampleDirection = sampleCosineWeightedHemisphere(r);
    sampleDirection = alignHemisphereWithNormal(sampleDirection, surfaceNormal);

    ray.origin = intersectionPoint + surfaceNormal * 1e-3f;
    ray.direction = sampleDirection;
    ray.color = color;
    ray.mask = RAY_MASK_SECONDARY;
}

kernel void shadowKernel(uint2 tid [[thread_position_in_grid]],
                         constant RTUniforms& uniforms,
                         device Ray* shadowRays,
                         device float* intersections,
                         texture2d<float, access::read> srcTex,
                         texture2d<float, access::write> dstTex) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) {
        return;
    }

    unsigned int rayIdx = tid.y * uniforms.width + tid.x;

    device Ray& shadowRay = shadowRays[rayIdx];
    float intersectionDistance = intersections[rayIdx];

    float3 color = srcTex.read(tid).xyz;

    if (shadowRay.maxDistance >= 0.0f && intersectionDistance < 0.0f) {
        color += shadowRay.color;
    }

    dstTex.write(float4(color, 1.0f), tid);
}

kernel void accumulateKernel(uint2 tid [[thread_position_in_grid]],
                             constant RTUniforms& uniforms,
                             texture2d<float> renderTex,
                             texture2d<float> prevTex,
                             texture2d<float, access::write> accumTex) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) {
        return;
    }

    float3 color = renderTex.read(tid).xyz;

    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(tid).xyz;
        prevColor *= uniforms.frameIndex;

        color += prevColor;
        color /= (uniforms.frameIndex + 1);
    }

    accumTex.write(float4(color, 1.0f), tid);
}
