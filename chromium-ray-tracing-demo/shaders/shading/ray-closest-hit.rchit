#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#pragma shader_stage(closest)
#pragma optionNV(fastmath on)
#pragma optionNV(ifcvt none)
#pragma optionNV(inline all)
#pragma optionNV(strict on)
#pragma optionNV(unroll all)

#define PI       3.141592653589793
#define HALF_PI  1.5707963267948966
#define TWO_PI   6.283185307179586
#define INV_PI   0.3183098861837907

#define GAMMA 2.2
#define INV_GAMMA 0.45454545454545453

#define EPSILON 0.001

#define LUMINANCE vec3(0.2126, 0.7152, 0.0722)

struct LightSource {
  vec4 emissionAndGeometryId;
  vec4 directionAndPdf;
};

struct RayPayload {
  vec4 radianceAndDistance;
  vec4 scatterDirectionAndPDF;
  vec3 throughput;
  uint seed;
  uint depth;
  bool shadowed;
};

struct ShadowRayPayload {
  vec3 hit;
  bool shadowed;
};

struct ShadingData {
  vec3 base_color;
  float metallic;
  float specular;
  float roughness;
  float clearcoat;
  float clearcoatGloss;
  float csw;
};

struct Vertex {
  vec4 position;
  vec4 normal;
  vec4 tangent;
  vec2 uv;
  vec2 pad_0;
};

struct Offset {
  uint face;
  uint vertex;
  uint material;
  uint pad_0;
};

struct Material {
  vec4 color;
  vec4 emission;
  float metalness;
  float roughness;
  float specular;
  float textureScaling;
  uint albedoIndex;
  uint normalIndex;
  uint emissionIndex;
  uint metalRoughnessIndex;
  float emissionIntensity;
  float metalnessIntensity;
  float roughnessIntensity;
  float pad_0;
};

struct Light {
  uint instanceIndex;
  float pad_0;
  float pad_1;
  float pad_2;
};

struct Instance {
  mat4x3 transformMatrix;
  vec4 padding_0;
  mat4x4 normalMatrix;
  uint vertexIndex;
  uint faceIndex;
  uint faceCount;
  uint materialIndex;
};

vec2 blerp(vec2 b, vec2 p1, vec2 p2, vec2 p3) {
  return (1.0 - b.x - b.y) * p1 + b.x * p2 + b.y * p3;
}

vec3 blerp(vec2 b, vec3 p1, vec3 p2, vec3 p3) {
  return (1.0 - b.x - b.y) * p1 + b.x * p2 + b.y * p3;
}

vec2 SampleTriangle(vec2 u) {
  float uxsqrt = sqrt(u.x);
  return vec2(1.0 - uxsqrt, u.y * uxsqrt);
}

// rand functions taken from neo java lib and
// https://github.com/nvpro-samples/optix_advanced_samples
const uint LCG_A = 1664525u;
const uint LCG_C = 1013904223u;
const int MAX_RAND = 0x7fff;
const int IEEE_ONE = 0x3f800000;
const int IEEE_MASK = 0x007fffff;

uint Tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;
  for (uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }
  return v0;
}

uint Rand(inout uint seed) { // random integer in the range [0, MAX_RAND]
  seed = 69069 * seed + 1;
  return ((seed = 69069 * seed + 1) & MAX_RAND);
}

float Randf01(inout uint seed) { // random number in the range [0.0f, 1.0f]
  seed = (LCG_A * seed + LCG_C);
  return float(seed & 0x00FFFFFF) / float(0x01000000u);
}

vec2 RandInUnitDisk(inout uint seed) {
  vec2 p = vec2(0);
  do {
    p = 2 * vec2(Randf01(seed), Randf01(seed)) - 1;
  } while (dot(p, p) >= 1);
  return p;
}

vec3 RandInUnitSphere(inout uint seed) {
  vec3 p = vec3(0);
  do {
    p = 2 * vec3(Randf01(seed), Randf01(seed), Randf01(seed)) - 1;
  } while (dot(p, p) >= 1);
  return p;
}

// source: internetz
vec3 Hash32(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * vec3(443.8975,397.2973, 491.1871));
  p3 += dot(p3, p3.yxz + 19.19);
  return fract(vec3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

vec3 DitherRGB(vec3 c, vec2 seed){
  return c + Hash32(seed) / 255.0;
}

float Luminance(vec3 color) {
  const vec3 luminance = { 0.30, 0.59, 0.11 };
  return dot(color, luminance);
}

vec3 SRGBToLinear(vec3 color) {
  return pow(color, vec3(INV_GAMMA));
}

vec3 Uncharted2ToneMapping(vec3 color) {
  float A = 0.15;
  float B = 0.50;
  float C = 0.10;
  float D = 0.20;
  float E = 0.02;
  float F = 0.30;
  float W = 11.2;
  float exposure = 2.0;
  color *= exposure;
  color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
  float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
  return SRGBToLinear(color / white);
}

vec3 FilmicToneMapping(vec3 color) {
  vec3 x = max(vec3(0.0), color - 0.004);
  color = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
  return SRGBToLinear(color);
}

float sqr(float f) {
  return f * f;
}

const float saturation = 0.12;
vec3 ColorGrading(vec3 color) {
  vec3 gray = vec3(dot(LUMINANCE, color));
  color = vec3(mix(color, gray, -saturation)) * 1.0;
  return color;
}

vec3 UniformSampleHemisphere(inout uint seed) {
  vec3 dir;
  float z = Randf01(seed);
  float r = sqrt(1.0 - z * z);
  float phi = TWO_PI * Randf01(seed);
  dir.x = r * cos(phi);
  dir.y = r * sin(phi);
  dir.z = z;
  return dir;
}

vec3 CosineSampleHemisphere(float u1, float u2) {
  vec3 dir;
  float r = sqrt(u1);
  float phi = TWO_PI * u2;
  dir.x = r * cos(phi);
  dir.y = r * sin(phi);
  dir.z = sqrt(max(0.0, 1.0 - dir.x*dir.x - dir.y*dir.y));
  return dir;
}

float powerHeuristic(float a, float b) {
  float t = a * a;
  return t / (b * b + t);
}

float GTR1(float NdotH, float a) {
  if (a >= 1.0) return INV_PI;
  float a2 = a * a;
  float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
  return (a2 - 1.0) / (PI * log(a2) * t);
}

float GTR2(float NdotH, float a) {
  float a2 = a * a;
  float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
  return a2 / (PI * t * t);
}

float SmithGGX_G(float NdotV, float a) {
  float a2 = a * a;
  float b = NdotV * NdotV;
  return 1.0 / (NdotV + sqrt(a2 + b - a2 * b));
}

float SchlickFresnelReflectance(float u) {
  float m = clamp(1.0 - u, 0.0, 1.0);
  float m2 = m * m;
  return m2 * m2 * m;
}

ShadingData shading;

// based on AMD baikal's disney implementation with some edits:
// https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRender-Baikal/blob/master/Baikal/Kernels/CL/disney.cl
float DisneyPdf(in const float NdotH, in const float NdotL, in const float HdotL) {
  const float d_pdf = NdotL * (1.0 / PI);
  const float r_pdf = GTR2(NdotH, shading.roughness) * NdotH / (4.0 * HdotL);
  const float c_pdf = GTR1(NdotH, 0.001) * NdotH / (4.0 * HdotL);
  return c_pdf * 0.001 + (shading.csw * r_pdf + (1.0 - shading.csw) * d_pdf);
}

vec3 DisneyEval(in float NdotL, in const float NdotV, in const float NdotH, in const float HdotL) {
  const vec3 cd_lin = shading.base_color;
  const vec3 c_spec0 = mix(shading.specular * vec3(0.3), cd_lin, shading.metallic);

  // Diffuse fresnel - go from 1 at normal incidence to 0.5 at grazing
  // and mix in diffuse retro-reflection based on roughness
  const float fwo = SchlickFresnelReflectance(NdotV);
  const float fwi = SchlickFresnelReflectance(NdotL);

  const float fd90 = 0.5 + 2.0 * HdotL * HdotL * shading.roughness;
  const float fd = mix(1.0, fd90, fwo) * mix(1.0, fd90, fwi);

  // Specular
  const float ds = GTR2(NdotH, shading.roughness);
  const float fh = SchlickFresnelReflectance(HdotL);
  const vec3 fs = mix(c_spec0, vec3(1), fh);

  float gs = 0.0;
  const float ro2 = sqr(shading.roughness * 0.5 + 0.5);
  gs = SmithGGX_G(NdotV, ro2);
  gs *= SmithGGX_G(NdotL, ro2);

  const float dr = GTR1(NdotH, 0.001);
  const float fr = mix(0.04, 1.0, fh);
  const float gr = SmithGGX_G(NdotV, 0.25) * SmithGGX_G(NdotL, 0.25);

  const vec3 f = (INV_PI * fd * cd_lin) * (1.0 - shading.metallic) + gs * fs * ds + 0.001 * gr * fr * dr;
  return f * NdotL;
}

vec4 DisneySample(inout uint seed, in const vec3 V, in const vec3 N) {
  float r1 = Randf01(seed);
  float r2 = Randf01(seed);

  const vec3 U = abs(N.z) < (1.0 - EPSILON) ? vec3(0, 0, 1) : vec3(1, 0, 0);
  const vec3 T = normalize(cross(U, N));
  const vec3 B = cross(N, T);

  // specular
  if (r2 < shading.csw) {
    r2 /= shading.csw;
    const float a = shading.roughness;
    const float cosTheta = sqrt((1.0 - r2) / (1.0 + (a*a-1.0) * r2));
    const float sinTheta = sqrt(max(0.0, 1.0 - (cosTheta * cosTheta)));
    const float phi = r1 * TWO_PI;
    vec3 H = normalize(vec3(
      cos(phi) * sinTheta,
      sin(phi) * sinTheta,
      cosTheta
    ));
    H = H.x * T + H.y * B + H.z * N;
    H = dot(H, V) <= 0.0 ? H * -1.0 : H;
    return vec4(reflect(-V, H), 1.0);
  }
  // diffuse
  r2 -= shading.csw;
  r2 /= (1.0 - shading.csw);
  const vec3 H = CosineSampleHemisphere(r1, r2);
  return vec4(T * H.x + B * H.y + N * H.z, 0.0);
}

struct HitAttributeData {
  vec2 bary;
};

hitAttributeEXT HitAttributeData Hit;

layout (location = 0) rayPayloadInEXT RayPayload Ray;
layout (location = 1) rayPayloadEXT ShadowRayPayload ShadowRay;

layout (binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout (binding = 3) uniform CameraBuffer {
  vec4 forward;
  mat4 viewInverse;
  mat4 projectionInverse;
  mat4 viewProjection;
  mat4 previousViewInverse;
  mat4 previousProjectionInverse;
  float aperture;
  float focusDistance;
  float zNear;
  float zFar;
} Camera;

layout (binding = 4) uniform SettingsBuffer {
  uint sampleCount;
  uint totalSampleCount;
  uint lightCount;
  uint screenWidth;
  uint screenHeight;
  uint envResolution;
  uint envTextureIndex;
  float envHdrMultiplier;
} Settings;

layout (binding = 5, std430) buffer AttributeBuffer {
  Vertex Vertices[];
};

layout (binding = 6, std430) buffer FaceBuffer {
  uint Faces[];
};

layout (binding = 7, std140, row_major) buffer InstanceBuffer {
  Instance Instances[];
};

layout (binding = 8, std430) buffer MaterialBuffer {
  Material Materials[];
};

layout (binding = 9, std430) buffer LightBuffer {
  Light Lights[];
};

layout (binding = 10) uniform sampler LinearSampler;
layout (binding = 11) uniform sampler NearestSampler;

layout (binding = 12) uniform texture2DArray TextureArray;
layout (binding = 13) uniform texture2DArray EnvironmentArray;
layout (binding = 14) uniform texture2DArray EnvMarginalArray;
layout (binding = 15) uniform texture2DArray EnvConditionalArray;

LightSource PickRandomLightSource(inout uint seed, in vec3 surfacePos, out vec3 lightDirection, out float lightDistance) {
  const uint lightIndex = uint(Randf01(seed) * Settings.lightCount);
  const uint geometryInstanceId = Lights[nonuniformEXT(lightIndex)].instanceIndex;
  const Instance instance = Instances[nonuniformEXT(geometryInstanceId)];

  const uint faceIndex = instance.faceIndex + uint(Randf01(seed) * instance.faceCount);

  const vec2 attribs = SampleTriangle(vec2(Randf01(seed), Randf01(seed)));

  const Vertex v0 = Vertices[instance.vertexIndex + Faces[faceIndex + 0]];
  const Vertex v1 = Vertices[instance.vertexIndex + Faces[faceIndex + 1]];
  const Vertex v2 = Vertices[instance.vertexIndex + Faces[faceIndex + 2]];

  const vec3 p0 = (instance.transformMatrix * vec4(v0.position.xyz, 1.0)).xyz;
  const vec3 p1 = (instance.transformMatrix * vec4(v1.position.xyz, 1.0)).xyz;
  const vec3 p2 = (instance.transformMatrix * vec4(v2.position.xyz, 1.0)).xyz;
  const vec3 pw = blerp(attribs, p0, p1, p2);

  const vec3 n0 = v0.normal.xyz;
  const vec3 n1 = v1.normal.xyz;
  const vec3 n2 = v2.normal.xyz;
  const vec3 nw = normalize(mat3x3(instance.normalMatrix) * blerp(attribs, n0.xyz, n1.xyz, n2.xyz));

  const float triangleArea = 0.5 * length(cross(p1 - p0, p2 - p0));

  const vec3 lightSurfacePos = pw;
  const vec3 lightEmission = Materials[instance.materialIndex].color.rgb;
  const vec3 lightNormal = normalize(lightSurfacePos - surfacePos);

  const vec3 lightPos = lightSurfacePos - surfacePos;
  const float lightDist = length(lightPos);
  const float lightDistSq = lightDist * lightDist;
  const vec3 lightDir = lightPos / lightDist;

  const float lightPdf = lightDistSq / (triangleArea * abs(dot(lightNormal, lightDir)));

  const vec4 emissionAndGeometryId = vec4(
    lightEmission, geometryInstanceId
  );
  const vec4 directionAndPdf = vec4(
    lightDir, lightPdf
  );

  lightDirection = lightDir;
  lightDistance = lightDist;

  return LightSource(
    emissionAndGeometryId,
    directionAndPdf
  );
}

vec4 EnvironmentSample(inout uint seed, inout vec3 color) {
  float r1 = Randf01(seed);
  float r2 = Randf01(seed);

  float v = texture(sampler2DArray(EnvMarginalArray, NearestSampler), vec3(r1, 0.0, Settings.envTextureIndex)).x;
  float u = texture(sampler2DArray(EnvConditionalArray, NearestSampler), vec3(r2, v, Settings.envTextureIndex)).x;

  color = texture(sampler2DArray(EnvironmentArray, LinearSampler), vec3(u, v, Settings.envTextureIndex)).rgb;
  float c_pdf = texture(sampler2DArray(EnvConditionalArray, NearestSampler), vec3(u, v, Settings.envTextureIndex)).y;
  float m_pdf = texture(sampler2DArray(EnvMarginalArray, NearestSampler), vec3(v, 0.0, Settings.envTextureIndex)).y;
  float pdf = c_pdf * m_pdf;

  float phi = u * TWO_PI;
  float theta = v * PI;

  if (sin(theta) == 0.0)
    pdf = 0.0;

  return vec4(
    -sin(theta) * cos(phi),
    cos(theta),
    -sin(theta)*sin(phi),
    (pdf * Settings.envResolution) / (2.0 * PI * PI * sin(theta))
  );
}

vec3 DirectLight(inout uint seed, const vec3 direction, const vec3 surfacePosition, const uint instanceId, in vec3 normal, in bool shadowed, in LightSource lightSource) {
  vec3 Lo = vec3(0.0);

  // sample environment map
  {
    vec3 color = vec3(0);
    vec4 bsdf = EnvironmentSample(seed, color);
    vec3 lightDir = bsdf.xyz;
    float lightDistance = 1000000.0;
    float lightPdf = bsdf.w;

    // shoot shadow ray
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 1, 0, 1, surfacePosition, EPSILON, lightDir, lightDistance - EPSILON, 1);

    // env map
    if (!ShadowRay.shadowed) {
      const vec3 N = normal;
      const vec3 V = -direction;
      const vec3 L = lightDir;
      const vec3 H = normalize(V + L);

      const float NdotH = abs(dot(N, H));
      const float NdotL = abs(dot(L, N));
      const float HdotL = abs(dot(H, L));
      const float NdotV = abs(dot(N, V));

      const float bsdfPdf = DisneyPdf(NdotH, NdotL, HdotL);
      const vec3 f = DisneyEval(NdotL, NdotV, NdotH, HdotL);

      const vec3 powerPdf = color * Settings.envHdrMultiplier;

      float misWeight = powerHeuristic(lightPdf, bsdfPdf);
      float cosWeight = abs(dot(lightDir, normal));
      if (misWeight > 0.0) {
        Lo += misWeight * f * cosWeight * powerPdf / max(0.001, lightPdf);
      }
    }
  }

  // lights
  {
    const vec4 directionAndPdf = lightSource.directionAndPdf;
    const vec4 emissionAndGeometryId = lightSource.emissionAndGeometryId;

    const vec3 lightEmission = emissionAndGeometryId.xyz;
    const uint lightGeometryInstanceId = uint(emissionAndGeometryId.w);

    // if we hit a light source, then just returns its emission directly
    if (instanceId == lightGeometryInstanceId) return lightEmission;

    // abort if we are occluded
    if (shadowed) return Lo;

    const vec3 lightDir = directionAndPdf.xyz;
    const float lightPdf = directionAndPdf.w;
    const vec3 powerPdf = lightEmission * Settings.lightCount;

    const vec3 N = normal;
    const vec3 V = -direction;
    const vec3 L = lightDir;
    const vec3 H = normalize(V + L);

    const float NdotH = max(0.0, dot(N, H));
    const float NdotL = max(0.0, dot(L, N));
    const float HdotL = max(0.0, dot(H, L));
    const float NdotV = max(0.0, dot(N, V));

    const float bsdfPdf = DisneyPdf(NdotH, NdotL, HdotL);

    const vec3 f = DisneyEval(NdotL, NdotV, NdotH, HdotL);

    const float misWeight = powerHeuristic(lightPdf, bsdfPdf);
    const float cosWeight = abs(dot(lightDir, normal));
    if (misWeight > 0.0) {
      Lo += misWeight * f * cosWeight * powerPdf / max(0.001, lightPdf);
    }
  }

  return max(vec3(0.0), Lo);
}

void main() {
  const vec3 surfacePosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_RayTmaxEXT;
  const vec3 rayDirection = gl_WorldRayDirectionEXT;

  const uint instanceId = gl_InstanceCustomIndexEXT;

  const Instance instance = Instances[nonuniformEXT(instanceId)];

  const Vertex v0 = Vertices[instance.vertexIndex + Faces[instance.faceIndex + gl_PrimitiveID * 3 + 0]];
  const Vertex v1 = Vertices[instance.vertexIndex + Faces[instance.faceIndex + gl_PrimitiveID * 3 + 1]];
  const Vertex v2 = Vertices[instance.vertexIndex + Faces[instance.faceIndex + gl_PrimitiveID * 3 + 2]];

  const vec2 u0 = v0.uv.xy, u1 = v1.uv.xy, u2 = v2.uv.xy;
  const vec3 n0 = v0.normal.xyz, n1 = v1.normal.xyz, n2 = v2.normal.xyz;
  const vec3 t0 = v0.tangent.xyz, t1 = v1.tangent.xyz, t2 = v2.tangent.xyz;

  const Material material = Materials[instance.materialIndex];

  const vec2 uv = blerp(Hit.bary.xy, u0.xy, u1.xy, u2.xy) * material.textureScaling;
  const vec3 no = blerp(Hit.bary.xy, n0.xyz, n1.xyz, n2.xyz);
  const vec3 ta = blerp(Hit.bary.xy, t0.xyz, t1.xyz, t2.xyz);

  const vec3 nw = normalize(mat3x3(instance.normalMatrix) * no);
  const vec3 tw = normalize(mat3x3(instance.normalMatrix) * ta);
  const vec3 bw = cross(nw, tw);

  const vec3 tex0 = texture(sampler2DArray(TextureArray, LinearSampler), vec3(uv, material.albedoIndex)).rgb;
  const vec3 tex1 = texture(sampler2DArray(TextureArray, LinearSampler), vec3(uv, material.normalIndex)).rgb;
  const vec3 tex2 = texture(sampler2DArray(TextureArray, LinearSampler), vec3(uv, material.metalRoughnessIndex)).rgb;
  const vec3 tex3 = texture(sampler2DArray(TextureArray, LinearSampler), vec3(uv, material.emissionIndex)).rgb;

  // material color
  vec3 color = tex0 + material.color.rgb;
  // material normal
  const vec3 normal = normalize(
    material.normalIndex > 0 ?
    mat3(tw, bw, nw) * normalize((pow(tex1, vec3(INV_GAMMA))) * 2.0 - 1.0).xyz :
    nw
  );
  // material metalness/roughness
  const vec2 metalRoughness = vec2(tex2.r, tex2.g);
  // material emission
  const vec3 emission = tex3 * material.emissionIntensity;

  uint seed = Ray.seed;
  float t = gl_HitTEXT;

  vec3 radiance = vec3(0);
  vec3 throughput = Ray.throughput.rgb;

  radiance += emission * throughput;

  shading.base_color = color;
  shading.metallic = clamp(metalRoughness.r + material.metalness, 0.001, 0.999) * material.metalnessIntensity;
  shading.specular = material.specular;
  shading.roughness = clamp(metalRoughness.g + material.roughness, 0.001, 0.999) * material.roughnessIntensity;
  {
    const vec3 cd_lin = shading.base_color;
    const float cd_lum = dot(cd_lin, vec3(0.3, 0.6, 0.1));
    const vec3 c_spec0 = mix(shading.specular * vec3(0.3), cd_lin, shading.metallic);
    const float cs_lum = dot(c_spec0, vec3(0.3, 0.6, 0.1));
    const float cs_w = cs_lum / (cs_lum + (1.0 - shading.metallic) * cd_lum);
    shading.csw = cs_w;
  }

  // pick a random light source
  // also returns a direction which we will shoot our shadow ray to
  vec3 lightDirection = vec3(0);
  float lightDistance = 0.0;
  LightSource lightSource = PickRandomLightSource(seed, surfacePosition, lightDirection, lightDistance);

  // shoot shadow ray
  traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 1, 0, 1, surfacePosition, EPSILON, lightDirection, lightDistance - EPSILON, 1);

  radiance += DirectLight(seed, rayDirection, surfacePosition, instanceId, normal, ShadowRay.shadowed, lightSource) * throughput;

  const vec4 bsdfDir = DisneySample(seed, -rayDirection, normal);

  const vec3 N = normal;
  const vec3 V = -rayDirection;
  const vec3 L = bsdfDir.xyz;
  const vec3 H = normalize(V + L);

  const float NdotH = abs(dot(N, H));
  const float NdotL = abs(dot(L, N));
  const float HdotL = abs(dot(H, L));
  const float NdotV = abs(dot(N, V));

  const float bsdfPdf = DisneyPdf(NdotH, NdotL, HdotL);
  if (bsdfPdf > 0.0) {
    throughput *= DisneyEval(NdotL, NdotV, NdotH, HdotL) / bsdfPdf;
  } else {
    t = -1.0;
  }

  Ray.radianceAndDistance = vec4(radiance, t);
  Ray.scatterDirectionAndPDF = vec4(bsdfDir.xyz, bsdfPdf);
  Ray.throughput = throughput;
  Ray.seed = seed;
}
