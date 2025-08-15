from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain, CMakeDeps

class LearnAI(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    exports = "CMakeLists.txt"

    def requirements(self):
        self.requires("libjpeg/9f", override=True)
        self.requires("openblas/0.3.29+ds-3", override=True)
        self.requires("opencv/4.10.0+dfsg-5", override=True)
        self.requires("eigen/3.4.0")
        
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        tc = CMakeToolchain(self)
        tc.generator = "Ninja"
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
