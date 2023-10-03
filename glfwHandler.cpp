// implementation of GLFWHandler
#include "glfwHandler.h"

GLFWHandler *GLFWHandler::instance = nullptr;

GLFWHandler::GLFWHandler()
{
    if (!glfwInit())
    {
        std::cout << "GLFW failed to initialize" << std::endl;
    }
}

GLFWHandler::~GLFWHandler()
{
    glfwTerminate();
}

GLFWHandler *GLFWHandler::getInstance()
{
    if (instance == nullptr)
    {
        instance = new GLFWHandler();
    }
    return instance;
}

GLFWwindow* GLFWHandler::getWindow()
{
    return window;
}

void GLFWHandler::initWindow(int width, int height, std::string title)
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, true);
    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    glfwGetWindowSize(window, &winSize.x, &winSize.y);

    SetCallbacks();
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glGenTextures(1, &fbTexture);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, nullptr);
}

void GLFWHandler::SetCallbacks()
{
    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double x, double y) {
        auto glfw = GLFWHandler::getInstance();
        glfw->mouseState.delta = owl::vec2f(x, y) - glfw->mouseState.position;
        glfw->mouseState.position = owl::vec2f(x, y);
    });

    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
        auto glfw = GLFWHandler::getInstance();
        if(glfw->mouseState.imGuiPolling)
            return;
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            glfw->mouseState.leftButtonDown = action == GLFW_PRESS;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            glfw->mouseState.rightButtonDown = action == GLFW_PRESS;
        }
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            glfw->mouseState.middleButtonDown = action == GLFW_PRESS;
        }
    });

    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto glfw = GLFWHandler::getInstance();
        if(action != GLFW_RELEASE)
        {
            glfw->key.keys[key] = action;
        }
        else
            glfw->key.keys.erase(key);
    });

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height) {
        auto glfw = GLFWHandler::getInstance();
        glfw->winSize = owl::vec2i(width, height);

        if (glfw->fbTexture == 0) 
        {
            GL_CHECK(glGenTextures(1, &(glfw->fbTexture)));
        }
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, glfw->fbTexture));
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                    GL_UNSIGNED_BYTE, nullptr));
        //glViewport(0, 0, width, height);
    });
}

void GLFWHandler::setWindowSize(int width, int height)
{
    glfwSetWindowSize(window, width, height);
}

void GLFWHandler::destroyWindow()
{
    glfwDestroyWindow(window);
}

void GLFWHandler::swapBuffers()
{
    glfwSwapBuffers(window);
}

void GLFWHandler::pollEvents()
{
    GLFWHandler::getInstance()->mouseState.delta = owl::vec2f(0, 0);
    glfwPollEvents();
}

int GLFWHandler::windowShouldClose()
{
    return glfwWindowShouldClose(window);
}

owl::vec2i GLFWHandler::getWindowSize()
{
    return winSize;
}

void *GLFWHandler::getWindowUserPointer()
{
    return glfwGetWindowUserPointer(window);
}

void GLFWHandler::draw(const void *fbPointer)
{
    glfwMakeContextCurrent(window);
    if (false)
    {
        (cudaGraphicsMapResources(1, &cuDisplayTexture));

        cudaArray_t array;
        (cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));
        {
            cudaMemcpy2DToArray(array,
                                0,
                                0,
                                reinterpret_cast<const void *>(fbPointer),
                                winSize.x * sizeof(uint32_t),
                                winSize.x * sizeof(uint32_t),
                                winSize.y,
                                cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
        GL_CHECK(glEnable(GL_TEXTURE_2D));
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0,
                         0, 0,
                         winSize.x, winSize.y,
                         GL_RGBA, GL_UNSIGNED_BYTE, fbPointer));
    }

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, winSize.x, winSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)winSize.x, 0.f, (float)winSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)winSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)winSize.x, (float)winSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)winSize.x, 0.f, 0.f);
    }
    glEnd();
    if (false)
    {
        (cudaGraphicsUnmapResources(1, &cuDisplayTexture));
    }
}