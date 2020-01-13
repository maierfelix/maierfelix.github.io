---
layout: post
title: Real-Time Ray-Tracing in WebGPU
gh-repo: maierfelix/webgpu
bigimg: /img/meetmat.png
tags: [WebGPU, RTX]
comments: true
---

## Intro

By the end of 2018, NVIDIA released the new GPU series Turing, best known for it's ability of accelerated Ray-Tracing.

Ray-Tracing is the process of simulating light paths from reality. In reality, billions of rays get shot around you and at some point, hit your eye. Up to today, simulating this process is one of the most expensive tasks in computer science and an ongoing research area.

Previously, if you were interested in RTX or wanted to learn about this topic, then you had a huge chunk of learning material in front of you.
Modern Graphics APIs became a lot more complicated to work with and RTX was only available for such APIs. You had to spend a lot time learning about them, before you could even start about the RTX topic itself.

**Note**: If you're not the owner of a RTX card, but have a GTX 1060+ around, then you are one of the lucky guys who can test RTX without the need to buy one of those more expensive cards.

## Luckily, there is WebGPU

WebGPU is the successor to WebGL and combines multiple Graphics APIs into one, standardized API. It is said, that WebGPU's API is a mixture of Apple's Metal API and parts of the Vulkan API, but a lot more easier to work with.

Most WebGPU implementations come with multiple rendering backends, such as D3D12, Vulkan, Metal and OpenGL. Depending on the user's setup, one of these backends get used, preferably the fastest one with the most reliability for the platform. The commands sent to WebGPU then get translated into one of these backends. 

## Upfront

Note that RTX is not available officially for WebGPU (yet?) and is only available for the [Node bindings for WebGPU](https://github.com/maierfelix/webgpu).
Recently I began adapting an unofficial Ray-Tracing extension for [Dawn](https://dawn.googlesource.com/dawn), which is the WebGPU implementation for [Chromium](https://www.chromium.org/). The Ray-Tracing extension is only implemented into the Vulkan backend so far, but a D3D12 implementation is on the Roadmap. You can find my Dawn Fork with Ray-Tracing capabilities [here](https://github.com/maierfelix/dawn-ray-tracing).

Now let me introduce you to the ideas and concepts of the Ray-Tracing extension.

### Bounding Volume Hierarchies

When dealing with Ray-Tracing, you often end up having to work with [Bounding Volume Hierarchies](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) (short: *"BVH"*). In Ray-Tracing, BVHs are used to encapsulate arbitrary geometry for faster Ray-Triangle intersection. The reason for this is, that for each Ray, you have to check all Triangles in a scene for intersection with the Ray at some point and this process quickly becomes expensive.

Previously, for Ray-Tracing projects, you had to implement your own BVH system. But today with RTX, you no longer have to do that, since the driver is generating the BVHs for you.

### Ray-Tracing Shaders

Previously, you only had Vertex, Fragment and Compute shaders, each specialized on their own task. The WebGPU Ray-Tracing extension exposes 3 new shader stages:

 - Ray-Generation (`.rgen`)
 - Ray-Closest-Hit (`.rchit`)
 - Ray-Miss (`.rmiss`)

##### Ray-Generation:
For each pixel on the screen, we want to create and shoot a Ray into a scene. This shader allows to generate and trace Rays into an Acceleration Container.

##### Ray-Closest-Hit:
A Ray can hit multiple surfaces (e.g. a Triangle), but often, we're only interested in the surface that is the closest to the Ray's origin. This shader gets invoked as soon as the closest hit surface, including information about the hit position and the relative object instance.

##### Ray-Miss:
This shader gets invoked, whenever a Ray didn't hit anything (e.g. "hit the sky").

### Shader Binding Table

The most common approach about shaders is to bind them, depending on how you want an object to look like on the Screen. In Ray-Tracing however, it's often impossible to known upfront which shaders to bind, and which shaders should get invoked in which order. To fix this, a new concept was introduced called the Shader Binding Table.

The Shader Binding Table's purpose is to batch shaders together into groups, and later, dynamically invoke them from the Ray-Tracing shaders, based on a Ray's tracing result (i.e. hit or miss a surface).


### Acceleration Containers

Acceleration Containers probably seem to be the most complicated thing at first, but they are actually fairly simple in their concept, once your got the idea.

There are two different kinds of Acceleration Containers:

1. Bottom-Level: The container stores references to geometry
2. Top-Level: The container stores instances with references to a Bottom-Level Container, each with an arbitrary transform

Generally said, a Bottom-Level Container contains just the meshes, while a Top-Level Containers describes, where to place these meshes in a virtual world. In fact, this process is similar to [Geometry Instancing](https://en.wikipedia.org/wiki/Geometry_instancing), a common approach in CG, which is about effectively reusing Geometry across a scene to reduce memory usage and improve performance.

## Coding Time

You can find a code reference [here](https://github.com/maierfelix/webgpu/blob/master/examples/ray-tracing/index.mjs).

After this Tutorial, you will be able to render this beautiful Triangle, fully Ray-Traced:

![](https://i.imgur.com/qSYWet5.png)

### Create Geometry

At first, you need some kind of geometry that you want to ray trace and draw to the screen.

We'll just use a simple Triangle at first:

````js
let triangleVertices = new Float32Array([
   1.0,  1.0, 0.0,
  -1.0,  1.0, 0.0,
   0.0, -1.0, 0.0
]);
// create a GPU local buffer containing the vertices
let triangleVertexBuffer = device.createBuffer({
  size: triangleVertices.byteLength,
  usage: GPUBufferUsage.COPY_DST
});
// upload the vertices to the GPU buffer
triangleVertexBuffer.setSubData(0, triangleVertices);
````

Note that the Ray-Tracing API strictly disallows using geometry buffers, which aren't uploaded on the GPU. The reason for this is, that performing Ray-Tracing on a mapped buffer is incredibly ineffective because of the synchronization.

Since it's always recommended to use an Index buffer aside your geometry, let's create one:
````js
let triangleIndices = new Uint32Array([
  0, 1, 2
]);
// create a GPU local buffer containing the indices
let triangleIndexBuffer = device.createBuffer({
  size: triangleIndices.byteLength,
  usage: GPUBufferUsage.COPY_DST
});
// upload the indices to the GPU buffer
triangleIndexBuffer.setSubData(0, triangleIndices);
````

### Create Bottom-Level Acceleration Container

Now we will create our first Acceleration Container, which stores a reference to the geometry we have just created:

````js
let geometryContainer = device.createRayTracingAccelerationContainer({
  level: "bottom",
  flags: GPURayTracingAccelerationContainerFlag.PREFER_FAST_TRACE,
  geometries: [
    {
      type: "triangles", // the geometry kind of the vertices (only triangles allowed)
      vertexBuffer: triangleVertexBuffer, // our GPU buffer containing the vertices
      vertexFormat: "float3", // one vertex is made up of 3 floats
      vertexStride: 3 * Float32Array.BYTES_PER_ELEMENT, // the byte stride between each vertex
      vertexCount: triangleVertices.length, // the total amount of vertices
      indexBuffer: triangleIndexBuffer, // (optional) the index buffer to use
      indexFormat: "uint32", // (optional) the format of the index buffer (Uint32Array)
      indexCount: triangleIndices.length // the total amount of indices
    }
  ]
});
````

### Create Top-Level Acceleration Container

This container will hold an instance with a reference to our triangle geometry. This instance defines how the geometry gets positioned in our world using the `transform` property. The property `geometryContainer` is used to assign geometry to the instance:

````js
let instanceContainer = device.createRayTracingAccelerationContainer({
  level: "top",
  flags: GPURayTracingAccelerationContainerFlag.PREFER_FAST_TRACE,
  instances: [
    {
      flags: GPURayTracingAccelerationInstanceFlag.TRIANGLE_CULL_DISABLE, // disable back-face culling
      mask: 0xFF, // in the shader, you can cull objects based on their mask
      instanceId: 0, // a custom Id which you can use to identify an object in the shaders
      instanceOffset: 0x0, // unused
      transform: { // defines how to position the instance in the world
        translation: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      },
      geometryContainer: geometryContainer // reference to a geometry container
    }
  ]
});
````

### Building Acceleration Containers

To let the driver build the BVHs and everything else for our Acceleration Containers, we use the Command `buildRayTracingAccelerationContainer`. One important thing to note here, is that the build order is important. Bottom-Level Containers must be built before Top-Level Containers, as they depend on each other.

````js
let commandEncoder = device.createCommandEncoder({});
commandEncoder.buildRayTracingAccelerationContainer(geometryContainer);
commandEncoder.buildRayTracingAccelerationContainer(instanceContainer);
queue.submit([ commandEncoder.finish() ]);
````

### Pixel Buffer

Ray-Tracing Shaders run similar as a Compute Shader and we somehow have to get our Ray-Traced Pixels to the Screen. To do so, we create a Pixel Buffer, which we will write our Pixels into, and then copy these Pixels to the screen.

````js
let pixelBufferSize = window.width * window.height * 4 * Float32Array.BYTES_PER_ELEMENT;
let pixelBuffer = device.createBuffer({
  size: pixelBufferSize,
  usage: GPUBufferUsage.STORAGE
});
````

### Bind Group Layout

There is a new type for bind group layouts, which allows us to assign a Top-Level Acceleration Container:

````js
let rtBindGroupLayout = device.createBindGroupLayout({
  bindings: [
    // the first binding will be the acceleration container
    {
      binding: 0,
      visibility: GPUShaderStage.RAY_GENERATION,
      type: "acceleration-container"
    },
    // the second binding will be the pixel buffer
    {
      binding: 1,
      visibility: GPUShaderStage.RAY_GENERATION,
      type: "storage-buffer"
    }
  ]
});
````

### Bind Group

When creating our bind group, we simply set the Acceleration Container and the Pixel Buffer. Note that for the Acceleration Container, we don't have to specify a size.

````js
let rtBindGroup = device.createBindGroup({
  layout: rtBindGroupLayout,
  bindings: [
    {
      binding: 0,
      accelerationContainer: instanceContainer,
      offset: 0,
      size: 0
    },
    {
      binding: 1,
      buffer: pixelBuffer,
      offset: 0,
      size: pixelBufferSize
    }
  ]
});
````

### Ray-Generation Shader

Ray-Tracing Shaders are similar to Compute Shaders, but when requiring the `GL_NV_ray_tracing` extension, things change quite a lot. I'll only describe the most important bits:

- `rayPayloadNV`: This payload is used to communicate between shader stages. For example, when the Hit-Shader is called, we can write a result into the payload, and read it back in the Ray-Generation Shader. Note that in other shaders, the payload is called `rayPayloadInNV`.
- `uniform accelerationStructureNV`: That's Top-Level Acceleration Container we've bound in our Bind Group.
- `gl_LaunchIDNV`: Is the relative Pixel Position, based on our `traceRays` call dimension.
- `gl_LaunchSizeNV`: Is the dimension, specified in our `traceRays` call.
- `traceNV`: Trace Rays into a Top-Level Acceleration Container

````glsl
#version 460
#extension GL_NV_ray_tracing : require
#pragma shader_stage(raygen)

layout(location = 0) rayPayloadNV vec3 hitValue;

layout(binding = 0, set = 0) uniform accelerationStructureNV container;

layout(std140, set = 0, binding = 1) buffer PixelBuffer {
  vec4 pixels[];
} pixelBuffer;

// see code reference on how to create this buffer
layout(set = 0, binding = 2) uniform Camera {
  mat4 view;
  mat4 projection;
} uCamera;

void main() {
  ivec2 ipos = ivec2(gl_LaunchIDNV.xy);
  const ivec2 resolution = ivec2(gl_LaunchSizeNV.xy);

  const vec2 offset = vec2(0);
  const vec2 pixel = vec2(ipos.x, ipos.y);
  const vec2 uv = (pixel / gl_LaunchSizeNV.xy) * 2.0 - 1.0;

  // create ray
  vec4 origin = uCamera.view * vec4(offset, 0, 1);
  vec4 target = uCamera.projection * (vec4(uv.x, uv.y, 1, 1));
  vec4 direction = uCamera.view * vec4(normalize(target.xyz), 0);

  // shoot ray into top-level container
  traceNV(
    container,           // top-level container
    gl_RayFlagsOpaqueNV, // additional flags
    0xFF,                // ignore mask
    0,                   // shader binding table group offset
    0,                   // shader binding table group offset
    0,                   // shader binding table group offset
    origin.xyz,          // ray origin
    0.01,                // minimum intersection range
    direction.xyz,       // ray direction
    4096.0,              // maximum intersection range
    0                    // payload location to use (see rayPayloadNV, rayPayloadInNV)
  );

  // write the pixel result into a pixel buffer
  const uint pixelIndex = ipos.y * resolution.x + ipos.x;
  pixelBuffer.pixels[pixelIndex] = vec4(hitValue, 1);
}
````

### Ray-Closest-Hit Shader

Gets executed for the closest intersection, relative to the Ray.

 - `rayPayloadInNV`: The Payload shared between this Shader and the Ray-Generation Shader.
 - `hitAttributeNV`: Intersection point defined in the Barycentric coordinate space.

Not used in this example, but important properties are also:

 - `gl_InstanceCustomIndexNV`: Returns us the Id of the instance we have intersected with - We can define the instance Id, when creating a Top-Level Acceleration Container.
 - `gl_WorldRayDirectionNV`: Returns the Ray's direction in World-Space and normalized.
 - `gl_WorldToObjectNV` and `gl_ObjectToWorldNV`: Can be used, to convert between World-Space and Object-Space. Note that both are `3x4` matrices.
 - `gl_HitTNV`: The traveled distance of the Ray.

Note that Hit-Shaders and the Miss-Shader all have the same properties available. You can find a list of all available properties [here](https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_ray_tracing.txt#L57-L109).

````glsl
#version 460
#extension GL_NV_ray_tracing : require
#pragma shader_stage(closest)

// shared with ray-gen shader
layout(location = 0) rayPayloadInNV vec3 hitValue;

hitAttributeNV vec2 attribs;

void main() {
  const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  hitValue = bary;
}
````

### Ray-Miss Shader

Gets executed whenever a Ray hits nothing at all.

````glsl
#version 460
#extension GL_NV_ray_tracing : require
#pragma shader_stage(miss)

// shared with ray-gen shader
layout(location = 0) rayPayloadInNV vec3 hitValue;

void main() {
  // return a greyish color when nothing is hit
  hitValue = vec3(0.15);
}
````

### Shader Binding Table

The Shader Binding Table has a fixed group order as well:

1. All Ray-Generation Shaders
2. All Ray-Hit Shaders (Closest-Hit, Any-Hit)
3. All Ray-Miss Shaders

To later index a shader (e.g. in `traceNV`), you simply use the offset of the shader relative to the shader's group:

````js
let shaderBindingTable = device.createRayTracingShaderBindingTable({
  shaders: [
    // group 0 (Gen)
    // offset 0
    {
      module: rayGenShaderModule,
      stage: GPUShaderStage.RAY_GENERATION
    },
    // group 1 (Hit)
    // offset 0
    {
      module: rayCHitShaderModule,
      stage: GPUShaderStage.RAY_CLOSEST_HIT
    },
    // group 2 (Miss)
    // offset 0
    {
      module: rayMissShaderModule,
      stage: GPUShaderStage.RAY_MISS
    }
  ]
});
````

### Ray-Tracing Pipeline

Creating the Ray-Tracing Pipeline is straightforward. The only important property here is `maxRecursionDepth`, which we can use to specify upfront how many Shader recursions we want to allow. Note that GPUs are bad at performing recursion, and when possible, recursive calls should be flattened into a loop. We will leave this value to `1`, so we can shoot our Rays in the Ray-Generation Shader.

````js
let rtPipeline = device.createRayTracingPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [rtBindGroupLayout]
  }),
  rayTracingState: {
    shaderBindingTable,
    maxRecursionDepth: 1
  }
});
````

### Tracing Rays

The Ray-Tracing Pass's only difference to other Passes is, that we have the Command `traceRays`, which allows us to shoot Rays with a given dimension. You might find it useful to kind of think of `traceRays` as the `dispatch` Command in a Compute Pass.

````js
let commandEncoder = device.createCommandEncoder({});
let passEncoder = commandEncoder.beginRayTracingPass({});
passEncoder.setPipeline(rtPipeline);
passEncoder.setBindGroup(0, rtBindGroup);
passEncoder.traceRays(window.width, window.height, 1);
passEncoder.endPass();
queue.submit([ commandEncoder.finish() ]);
````

### Blit to Screen

I left the part of setting up the Rasterization pipeline, as it should be straightforward and only the Blit Shaders should be interesting.

The blit vertex Shader is created like this and defines a Full-Screen Quad:
````glsl
#version 450
#pragma shader_stage(vertex)

layout (location = 0) out vec2 uv;

void main() {
  vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
  uv = pos;
}
````

Note that you have to draw with a Vertex Count `3` to successfully run this shader. Also, make sure that your Blit Pass has a color attachment with the Swapchain Image as Input.

````glsl
#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) buffer PixelBuffer {
  vec4 pixels[];
} pixelBuffer;

layout(set = 0, binding = 1) uniform ScreenDimension {
  vec2 resolution;
};

void main() {
  const ivec2 bufferCoord = ivec2(floor(uv * resolution));
  const vec2 fragCoord = (uv * resolution);
  const uint pixelIndex = bufferCoord.y * uint(resolution.x) + bufferCoord.x;

  vec4 pixelColor = pixelBuffer.pixels[pixelIndex];
  outColor = pixelColor;
}
````

The fragment Shader is just copying the pixels of the Pixel Buffer into the Color Attachment. 
