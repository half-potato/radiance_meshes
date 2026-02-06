import slangtorch
import os

class ShaderManager:
    def __init__(self, shaders_dir):
        self.shaders_dir = shaders_dir
        self._cache = {}
        
        # Load static shaders immediately or lazily as preferred
        # Since these don't depend on defines, we can load them once to fail fast on errors
        self.vertex_shader = slangtorch.loadModule(os.path.join(shaders_dir, "vertex_shader.slang"))
        self.tile_shader = slangtorch.loadModule(os.path.join(shaders_dir, "tile_shader.slang"))

    def get_alphablend(self, tile_height, tile_width, aux_dim):
        key = (tile_height, tile_width, aux_dim, "alphablend")
        if key not in self._cache:
            defines = {
                "PYTHON_TILE_HEIGHT": tile_height,
                "PYTHON_TILE_WIDTH": tile_width,
                "PYTHON_AUX_DIM": aux_dim
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "alphablend_shader.slang"),
                defines=defines
            )
        return self._cache[key]

    def get_interp(self, tile_height, tile_width, aux_dim):
        key = (tile_height, tile_width, aux_dim, "interp")
        if key not in self._cache:
            defines = {
                "PYTHON_TILE_HEIGHT": tile_height,
                "PYTHON_TILE_WIDTH": tile_width,
                "PYTHON_AUX_DIM": aux_dim
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "alphablend_shader_interp.slang"),
                defines=defines
            )
        return self._cache[key]

# Usage instance
shaders_path = os.path.dirname(__file__)
shader_manager = ShaderManager(shaders_path)
