#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 outColor;

layout (binding = 0, std140) buffer PixelBuffer {
  vec4 pixels[];
} pixelBuffer;

layout (binding = 1) uniform SettingsBuffer {
  uint sampleCount;
  uint totalSampleCount;
  uint lightCount;
  uint screenWidth;
  uint screenHeight;
  uint envResolution;
  uint envTextureIndex;
  float envHdrMultiplier;
} Settings;

vec4 SamplePixelBuffer(vec2 uv) {
  const vec2 resolution = vec2(Settings.screenWidth, Settings.screenHeight);
  const ivec2 bufferCoord = ivec2(floor(uv * resolution));
  const uint pixelIndex = bufferCoord.y * uint(resolution.x) + bufferCoord.x;
  return pixelBuffer.pixels[pixelIndex];
}

void main() {
  vec4 pixelColor = SamplePixelBuffer(uv);

  // effect settings
  float vignetteIntensity = 0.41;

  // vignette effect
  {
    vec2 pos = uv.xy - vec2(0.5);
    float r = dot(pos, pos) * vignetteIntensity + 1.0;
    pixelColor /= r * r;
  }

  outColor = vec4(pixelColor.rgb, 1.0);
}
