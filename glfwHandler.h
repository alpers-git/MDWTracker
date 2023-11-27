#pragma once

//A singleton class that handles all GLFW related functions
#include <iostream>
#include <string>
#include <unordered_map>

#ifdef __linux__ 
#include "GL/gl.h"
#endif
#include "GLFW/glfw3.h"
#include "owl/common/math/vec.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "owl/helper/cuda.h"

//#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK( call )                                            \
    do                                                                  \
      {                                                                 \
        call;                                                           \
        GLenum err = glGetError();                                      \
        if( err != GL_NO_ERROR )                                        \
          {                                                             \
            std::stringstream ss;                                       \
            ss << "GL error " <<  getGLErrorString( err ) << " at "     \
               << __FILE__  << "(" <<  __LINE__  << "): " << #call      \
               << std::endl;                                            \
            std::cerr << ss.str() << std::endl;                         \
            throw std::runtime_error( ss.str().c_str() );               \
          }                                                             \
      }                                                                 \
    while (0)


#    define GL_CHECK_ERRORS( )                                          \
    do                                                                  \
      {                                                                 \
        GLenum err = glGetError();                                      \
        if( err != GL_NO_ERROR )                                        \
          {                                                             \
            std::stringstream ss;                                       \
            ss << "GL error " <<  getGLErrorString( err ) << " at "     \
               << __FILE__  << "(" <<  __LINE__  << ")";                \
            std::cerr << ss.str() << std::endl;                         \
            throw std::runtime_error( ss.str().c_str() );               \
          }                                                             \
      }                                                                 \
    while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif


class GLFWHandler
{
    public:
        static GLFWHandler* getInstance();
        GLFWwindow* getWindow();
        void initWindow(int width, int height, std::string title);
        void destroyWindow();
        void swapBuffers();
        void pollEvents();
        int windowShouldClose();
        void* getWindowUserPointer();
        owl::vec2i getWindowSize();
        void draw(const void* fbpointer);

        void setWindowSize(int width, int height);


        struct MouseState {
            owl::vec2f position;
            owl::vec2f delta;

            bool imGuiPolling = false;
            bool leftButtonDown = false;
            bool rightButtonDown = false;
            bool middleButtonDown = false;
        } mouseState;

        struct KeyboardState {
            std::unordered_map<int, int> keys;// Keys and their last GLFW action

            // Returns true if the key is currently down
            bool isDown(int key) {
                if (keys.find(key) == keys.end()) 
                {
                    return false;
                }
                return true; 
            }

            // Returns true if the key was pressed this frame
            bool isPressed(int key) {
                auto foundKey = keys.find(key);
                if (foundKey == keys.end()) 
                {
                    return false;
                }
                bool rtn = (foundKey->second == GLFW_PRESS);//to avoid returning repeated keys
                foundKey->second = GLFW_REPEAT;
                return rtn;
            }

            // Returns true if the key was held down multiple frames
            bool isRepeated(int key) {
                auto foundKey = keys.find(key);
                if (foundKey == keys.end()) 
                {
                    return false;
                }
                return foundKey->second == GLFW_REPEAT;
            }
        } key;
    private:
        GLFWwindow* window;
        owl::vec2i winSize;

        GLuint   fbTexture  {0};
        cudaGraphicsResource_t cuDisplayTexture { 0 };

        static GLFWHandler* instance;
        GLFWHandler();
        ~GLFWHandler();


        void SetCallbacks();     
};