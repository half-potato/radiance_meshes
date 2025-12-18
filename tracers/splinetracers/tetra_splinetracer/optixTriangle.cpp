//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#define NDEBUG 1

#include <glad/glad.h> // Needs to be included before gl_interop
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include "glm/glm.hpp"
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/vec_math.h>

#include "optixTriangle.h"
#include "TriangleMesh.h"
#include "CUDABuffer.h"
#include "Forward.h"
#include "GAS.h"
#include "ply_file_loader.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include <glm/gtx/string_cast.hpp>

#include <chrono>
using namespace std::chrono;

const float C0 = 0.28209479177387814;

/*
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;

struct HitGroupData {
    GAppearance *apps;
    size_t n1;
    GTransform *transforms;
    size_t n2;
};

typedef SbtRecord<HitGroupData> HitGroupSbtRecord;
*/

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 2.0f, 0.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 3.0f, 1.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}
//
//
//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;

Params*  d_params = nullptr;
Params   params   = {};
int32_t                 width    = 800;
int32_t                 height   = 800;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    width   = res_x;
    height  = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

// ====================================================================
void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    // params.cam_eye = camera.eye();
    // camera.UVWFrame( params.cam_u, params.cam_v, params.cam_w );
    /*
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        */

}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );

    // Realloc accumulation buffer
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    // CUDA_CHECK( cudaMalloc(
    //             reinterpret_cast<void**>( &params.accum_buffer ),
    //             width*height*sizeof(float4)
    //             ) );
}
void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    handleCameraUpdate( params );
    handleResize( output_buffer );
}

void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


int main( int argc, char* argv[] )
{
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        TriangleMesh model;

        //*
        gspline::GaussianScene dscene = *gspline::ReadSceneFromFile(
                "/usr/local/google/home/alexmai/optix-examples/models/lego.ply", false);
        gspline::GaussianScene *scene = &dscene;

        float s = 2;
        size_t numPrimitives = scene->means.size();
        // size_t numPrimitives = 1000;
        std::vector<GTransform> transforms(numPrimitives);
        for (int i = 0; i < numPrimitives; i++) {
            glm::vec4 quat = {scene->rotations[i][0], scene->rotations[i][1],
                           scene->rotations[i][2], scene->rotations[i][3]};
            quat = glm::normalize(quat);
            // float scale = s*fmin(fmin(scene->scales[i][0], scene->scales[i][1]), scene->scales[i][2]);
            // float scale = powf(scene->scales[i][0]*scene->scales[i][1]*scene->scales[i][2], 1.f/3.f);
            float alpha = scene->alphas[i];
            glm::vec3 center = {scene->means[i][0], scene->means[i][1], scene->means[i][2]};
            glm::vec3 size = {
                fmax(s*scene->scales[i][0], 1e-3),
                fmax(s*scene->scales[i][1], 1e-3),
                fmax(s*scene->scales[i][2], 1e-3),
            };
            glm::vec3 color = {fmin(fmax(scene->spherical_harmonics[i][0] * C0 + 0.5, 0.0), 1.f),
                          fmin(fmax(scene->spherical_harmonics[i][1] * C0 + 0.5, 0.0), 1.f),
                          fmin(fmax(scene->spherical_harmonics[i][2] * C0 + 0.5, 0.0), 1.f)};

            const float r = quat.x;
            const float x = quat.y;
            const float y = quat.z;
            const float z = quat.w;

            const glm::mat3 Rt = {
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - r * z),
                2.0 * (x * z + r * y),

                2.0 * (x * y + r * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - r * x),

                2.0 * (x * z - r * y),
                2.0 * (y * z + r * x),
                1.0 - 2.0 * (x * x + y * y)
            };
            const glm::mat3 R = glm::transpose(Rt);
            //
            glm::mat3 S = {
              size.x, 0, 0,
              0, size.y, 0,
              0, 0, size.z,
            };
            glm::mat3 T = Rt * S;
            // std::cout << glm::to_string(T) << std::endl;
            glm::mat4x3 xfm;

            xfm[0] = T[0];
            xfm[1] = T[1];
            xfm[2] = T[2];
            xfm[3] = center;

            float density = alpha = 1000.f*fmax(-log(1-alpha), 0.f);
            // if (fmin(size.x, fmin(size.y, size.z)) < 1e-4 || density < 0.001) {
            //     continue;
            // }
            model.addOctehedra(xfm, density, color);
            transforms[i] = {
                .scales = {size.x, size.y, size.z},
                .mean = {center.x, center.y, center.z},
                .quat = {r, x, y, z},
            };
        }
        //*/
        /*
        std::vector<GTransform> transforms;

        float s = 1e-2;
        glm::vec3 size1 = {s, s, s};
        glm::vec3 center1 = {0.01f, 0.1f, 0.0f};
        glm::mat4x3 xfm1;

        glm::vec4 quat = {0.3685775126341173, 0.5154692800079472, 0.7305244649761721, 0.25451138406722235};
        // glm::vec4 quat = {1, 0.1, 0, 0};
        quat = glm::normalize(quat);
        // glm::vec4 quat = {1, 0, 0, 0};
        const float r = quat.x;
        const float x = quat.y;
        const float y = quat.z;
        const float z = quat.w;

        const glm::mat3 Rt = {
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - r * z),
            2.0 * (x * z + r * y),

            2.0 * (x * y + r * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - r * x),

            2.0 * (x * z - r * y),
            2.0 * (y * z + r * x),
            1.0 - 2.0 * (x * x + y * y)
        };
        const glm::mat3 R = glm::transpose(Rt);
        //
        glm::mat3 S = {
          size1.x, 0, 0,
          0, size1.y, 0,
          0, 0, size1.z,
        };
        glm::mat3 T = Rt * S;
        // std::cout << glm::to_string(T) << std::endl;

        xfm1[0] = T[0];
        xfm1[1] = T[1];
        xfm1[2] = T[2];
        xfm1[3] = center1;

        model.addOctehedra(xfm1, 1000.f, glm::vec3(0.0f, 0.0f, 1.f));
        transforms.push_back({
            .scales = {size1.x, size1.y, size1.z},
            .mean = {center1.x, center1.y, center1.z},
            // .quat = {r, x, y, z},
            .quat = {r, x, y, z},
        });

        glm::vec3 size2 = {s, s, s};
        glm::vec3 center2 = {0.0100001f, 0.1f, 0.5f};
        glm::mat4x3 xfm2;

        xfm2[0] = T[0];
        xfm2[1] = T[1];
        xfm2[2] = T[2];
        xfm2[3] = center2;

        // model.addOctehedra(xfm2, 100.f, glm::vec3(1.0f, 0.3f, 0.1f));
        // transforms.push_back({
        //     .scales = {size2.x, size2.y, size2.z},
        //     .mean = {center2.x, center2.y, center2.z},
        //     .quat = {r, x, y, z},
        // });
        //*/

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        const uint8_t device = 0;
        GAS gas (context, device, model);
        Forward forward (context, device, model, transforms);

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        CUstream stream;
        CUDA_CHECK( cudaStreamCreate( &stream ) );

        configureCamera( camera, width, height );

        if( outfile.empty() ) {
            
            GLFWwindow* window = sutil::initUI( "optixMeshViewer", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &params               );

            trackball.setCamera( &camera );
            trackball.setMoveSpeed( 10.0f );
            trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
            trackball.setGimbalLock(true);

            // sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time( 0.0 );
            std::chrono::duration<double> render_time( 0.0 );
            std::chrono::duration<double> display_time( 0.0 );

            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                updateState( output_buffer, params );
                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                t1 = std::chrono::steady_clock::now();

                float3 U, V, W;
                camera.UVWFrame(U, V, W);
                std::vector<float3> ray_origins(width*height);
                std::vector<float3> ray_directions(width*height);
                for (int idx_x=0; idx_x<width; idx_x++) {
                    for (int idx_y=0; idx_y<height; idx_y++) {
                        const float2 d = {2.f * idx_x / height - 1.f, 2.f * idx_y / height - 1.f};
                        const float3 direction   = normalize( d.x * U + d.y * V + W );
                        ray_origins[idx_y * width + idx_x] = camera.eye();
                        ray_directions[idx_y * width + idx_x] = direction;
                    }
                }
                CUDABuffer rayo_buffer, rayd_buffer;
                rayo_buffer.alloc_and_upload(ray_origins);
                rayd_buffer.alloc_and_upload(ray_directions);

                auto start = high_resolution_clock::now();
                forward.trace_rays(
                    gas.gas_handle,
                    width * height,
                    (float3 *)rayo_buffer.d_pointer(),
                    (float3 *)rayd_buffer.d_pointer(),
                    output_buffer.map());

                output_buffer.unmap();
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start);
                float ms = float(duration.count()) / 1000.f;
                // std::cout << duration.count() << std::endl;
                // printf("Rendered in %f ms. Frame rate: %f\n", ms, 1000.f/ms);
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe( output_buffer, gl_display, window );
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats( state_update_time, render_time, display_time );

                glfwSwapBuffers(window);

                // ++params.subframe_index;
            }
            while( !glfwWindowShouldClose( window ) );
            CUDA_SYNC_CHECK();
            sutil::cleanupUI( window );
        // } else {
        //     CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
        //     CUDA_CHECK( cudaMemcpy(
        //                 reinterpret_cast<void*>( d_param ),
        //                 &params, sizeof( params ),
        //                 cudaMemcpyHostToDevice
        //                 ) );
        //
        //     auto start = high_resolution_clock::now();
        //     OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
        //
        //     CUDA_SYNC_CHECK();
        //     output_buffer.unmap();
        //
        //     // Time
        //     auto stop = high_resolution_clock::now();
        //     auto duration = duration_cast<microseconds>(stop - start);
        //     float ms = float(duration.count()) / 1000.f;
        //     std::cout << duration.count() << std::endl;
        //     printf("Rendered in %f ms. Frame rate: %f\n", ms, 1000.f/ms);
        //
        //     sutil::ImageBuffer buffer;
        //     buffer.data         = output_buffer.getHostPointer();
        //     buffer.width        = width;
        //     buffer.height       = height;
        //     buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        //     sutil::saveImage( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
