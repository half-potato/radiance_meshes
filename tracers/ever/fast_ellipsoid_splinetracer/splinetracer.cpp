// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/vec_math.h>

#include "CUDABuffer.h"
#include "Forward.h"
#include "GAS.h"
#include "structs.h"
#include "create_aabbs.h"
#include "sh.h"

#include "ply_file_loader.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>


#include <chrono>
using namespace std::chrono;

const float C0 = 0.28209479177387814;

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 2.0f, 0.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, -3.0f, -1.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}
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
int32_t                 width    = 1280;
int32_t                 height   = 720;
float fx = 301.9503085493233;
float fy = 302.21733327626583;
double lastTime = glfwGetTime();


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
    if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
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

static void clear_key(GLFWwindow* window, int32_t key) {
    for (int i=0; i<20; i++) {
        glfwGetKey( window, key );
    }
}

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    float3 right, up, forward;
    camera.UVWFrame(right, up, forward);
	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float dt = 0.5 * fmax(float(currentTime - lastTime), 0.f);

    if (glfwGetKey( window, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS) {
        dt *= 10;
    }

    if (glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS) {
        camera.setEye(camera.eye() + dt*forward);
        camera.setLookat(camera.lookat() + dt*forward);
        camera_changed = true;
        printf("w");
    }

    if (glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS) {
        camera.setEye(camera.eye() - dt*forward);
        camera.setLookat(camera.lookat() - dt*forward);
        camera_changed = true;
        printf("s");
    }

    if (glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS) {
        camera.setEye(camera.eye() + dt*right);
        camera.setLookat(camera.lookat() + dt*right);
        camera_changed = true;
        printf("d");
    }

    if (glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS) {
        camera.setEye(camera.eye() - dt*right);
        camera.setLookat(camera.lookat() - dt*right);
        camera_changed = true;
        printf("a");
    }

    if (glfwGetKey( window, GLFW_KEY_E ) == GLFW_PRESS) {
        camera.setEye(camera.eye() + dt*up);
        camera.setLookat(camera.lookat() + dt*up);
        camera_changed = true;
    }

    if (glfwGetKey( window, GLFW_KEY_Q ) == GLFW_PRESS) {
        camera.setEye(camera.eye() - dt*up);
        camera.setLookat(camera.lookat() - dt*up);
        camera_changed = true;
    }

    clear_key(window, GLFW_KEY_Q);
    clear_key(window, GLFW_KEY_E);
    clear_key(window, GLFW_KEY_W);
    clear_key(window, GLFW_KEY_A);
    clear_key(window, GLFW_KEY_S);
    clear_key(window, GLFW_KEY_D);
    clear_key(window, GLFW_KEY_LEFT_SHIFT);

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
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );
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

float sigmoid(float x) {
    return 1/ (1+exp(-x));
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    std::cerr << "         --fisheye      sets whether to use fisheye\n";
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
    bool fisheye = false;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--fisheye")
        {
            fisheye = true;
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
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        Primitives model;

        gspline::GaussianScene dscene = gspline::ReadSceneFromFile(outfile, false);
        gspline::GaussianScene *scene = &dscene;

        size_t numPrimitives = scene->means.size();

        std::vector<glm::vec3> means(numPrimitives);
        std::vector<glm::vec3> scales(numPrimitives);
        std::vector<glm::vec4> quats(numPrimitives);
        std::vector<float> densities(numPrimitives);
        std::vector<glm::vec3> colors(numPrimitives);
        auto deg = 0;
        auto feature_size = (deg+1)*(deg+1);
        std::vector<float> features(numPrimitives * feature_size * 3);

        float s = 1;
        for (int i = 0; i < numPrimitives; i++) {
            glm::vec4 quat = {scene->rotations[i][0], scene->rotations[i][1],
                           scene->rotations[i][2], scene->rotations[i][3]};
            quat = glm::normalize(quat);
            float alpha = scene->alphas[i];
            glm::vec3 center = {scene->means[i][0], scene->means[i][1], scene->means[i][2]};
            glm::vec3 size = {
                scene->scales[i][0],
                scene->scales[i][1],
                scene->scales[i][2],
            };
            glm::vec3 color = {scene->spherical_harmonics[i][0],
                          scene->spherical_harmonics[i][1],
                          scene->spherical_harmonics[i][2]};

            float density = alpha;

            means[i] = center;
            scales[i] = size;
            quats[i] = quat;
            densities[i] = density;
            colors[i] = color;
            for (int j=0; j<3; j++) {
                float v = scene->spherical_harmonics[i][3 * 0 + j];
                int start_ind = i * feature_size * 3;
                features[start_ind + 3 * 0 + j] = v;
                for (int k=1; k<feature_size; k++) {
                    features[start_ind + 3 * k + j] = scene->spherical_harmonics[i][3 * k + j];
                }
            }
        }
        CUDABuffer meansBuffer, scalesBuffer, quatsBuffer, densitiesBuffer, colorsBuffer;
        meansBuffer.alloc_and_upload(means);
        scalesBuffer.alloc_and_upload(scales);
        quatsBuffer.alloc_and_upload(quats);
        densitiesBuffer.alloc_and_upload(densities);
        colorsBuffer.alloc_and_upload(features);

        model.means = (float3 *)meansBuffer.d_pointer();
        model.scales = (float3 *)scalesBuffer.d_pointer();
        model.quats = (float4 *)quatsBuffer.d_pointer();
        model.densities = (float *)densitiesBuffer.d_pointer();
        model.features = (float *)colorsBuffer.d_pointer();
        model.feature_size = 3;
        model.num_prims = numPrimitives;
        model.prev_alloc_size = 0;
        create_aabbs(model);

        std::cout << "done" << std::endl;

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
        GAS gas (context, device, model, true);
        Forward forward (context, device, model, false);
        float4 *initial_drgb;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&initial_drgb),
                              width * height * sizeof(float4)));

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        CUstream stream;
        CUDA_CHECK( cudaStreamCreate( &stream ) );

        configureCamera( camera, width, height );

        if( true ) {
            
            GLFWwindow* window = sutil::initUI( "optixMeshViewer", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &params               );

            trackball.setCamera( &camera );
            trackball.setMoveSpeed( -5.0f );
            trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
            trackball.setGimbalLock(false);

            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time( 0.0 );
            std::chrono::duration<double> render_time( 0.0 );
            std::chrono::duration<double> display_time( 0.0 );

            CUDABuffer rayo_buffer, rayd_buffer;
            std::vector<float3> ray_origins(width*height);
            std::vector<float3> ray_directions(width*height);
            rayo_buffer.alloc_and_upload(ray_origins);
            rayd_buffer.alloc_and_upload(ray_directions);
            do
            {
                glfwPollEvents();

                auto t1 = std::chrono::steady_clock::now();
                updateState( output_buffer, params );
                auto t0 = std::chrono::steady_clock::now();


                float3 U, V, W;
                camera.UVWFrame(U, V, W);

                Cam cam = {
                    .fx = fx,
                    .fy = fy,
                    .height = height,
                    .width = width,
                    .U = U,
                    .V = V,
                    .W = W,
                    .eye = {camera.eye().x, camera.eye().y, camera.eye().z}
                };


                eval_sh(model, deg, 3*(deg+1)*(deg+1),
                        {camera.eye().x, camera.eye().y, camera.eye().z},
                        (float *)colorsBuffer.d_pointer());
                model.features = (float *)colorsBuffer.d_pointer();

                state_update_time += t0 - t1;

                t1 = std::chrono::steady_clock::now();
                t0 = t1;
                auto start = high_resolution_clock::now();
                forward.trace_rays(
                    gas.gas_handle,
                    width * height,
                    (float3 *)rayo_buffer.d_pointer(),
                    (float3 *)rayd_buffer.d_pointer(),
                    output_buffer.map(),
                    0, 0.0, 1e7, initial_drgb, &cam, 100);
                lastTime = glfwGetTime();
                output_buffer.unmap();
                t1 = std::chrono::steady_clock::now();
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start);
                float ms = float(duration.count()) / 1000.f;
                std::cout << std::flush;
                CUDA_SYNC_CHECK();
                render_time += t1 - t0;
                t0 = t1;

                t1 = std::chrono::steady_clock::now();
                displaySubframe( output_buffer, gl_display, window );
                t0 = std::chrono::steady_clock::now();
                display_time = t0-t1 ;

                sutil::displayStats( state_update_time, render_time, display_time );

                glfwSwapBuffers(window);

            }
            while( !glfwWindowShouldClose( window ) );
            CUDA_SYNC_CHECK();
            sutil::cleanupUI( window );
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(initial_drgb)));
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
