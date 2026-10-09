import slangtorch
import os

# Recentered ray-tet intersection (fixes f32 cancellation in
# t_exit - t_enter that zeroes alpha/gradients on thin crossings).
# Default ON; set DSPLAT_RECENTER_INTERSECT=0 to get the legacy
# camera-origin math (bit-identical to pre-fix) for A/B or debugging.
RECENTER_INTERSECT = os.environ.get("DSPLAT_RECENTER_INTERSECT", "1") == "1"

def _numerics_defines():
    return {"RECENTER_INTERSECT": 1} if RECENTER_INTERSECT else {}

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
                "PYTHON_AUX_DIM": aux_dim,
                **_numerics_defines()
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
                "PYTHON_AUX_DIM": aux_dim,
                **_numerics_defines()
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "alphablend_shader_interp.slang"),
                defines=defines
            )
        return self._cache[key]

    def get_shared_ray_flash(self, tile_height, tile_width, aux_dim, material_mode=0, component=0):
        if aux_dim < 8:
            raise ValueError("Shared-ray flash requires roughness and diffuse albedo channels")
        if material_mode not in range(4) or component not in range(4):
            raise ValueError("Invalid flash material/component selector")
        if material_mode and aux_dim < 13:
            raise ValueError("Flash material requires five extra auxiliary channels")
        key = (tile_height, tile_width, aux_dim, "shared_ray_flash", material_mode, component)
        if key not in self._cache:
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "shared_ray_flash.slang"),
                defines={"PYTHON_TILE_HEIGHT": tile_height, "PYTHON_TILE_WIDTH": tile_width,
                         "PYTHON_AUX_DIM": aux_dim, "FLASH_MATERIAL_MODE": material_mode,
                         "FLASH_COMPONENT": component, **_numerics_defines()})
        return self._cache[key]

    def get_interval(self, tile_height, tile_width, aux_dim):
        key = (tile_height, tile_width, aux_dim, "interval")
        if key not in self._cache:
            defines = {
                "PYTHON_TILE_HEIGHT": tile_height,
                "PYTHON_TILE_WIDTH": tile_width,
                "PYTHON_AUX_DIM": aux_dim,
                **_numerics_defines()
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "alphablend_shader_interval.slang"),
                defines=defines
            )
        return self._cache[key]

    def get_interval_generate(self, aux_dim):
        key = (aux_dim, "interval_generate")
        if key not in self._cache:
            defines = {
                "PYTHON_AUX_DIM": aux_dim,
                **_numerics_defines()
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "interval_generate.slang"),
                defines=defines
            )
        return self._cache[key]

    def get_raytrace(self, aux_dim):
        key = (aux_dim, "raytrace")
        if key not in self._cache:
            defines = {
                "PYTHON_AUX_DIM": aux_dim,
                **_numerics_defines()
            }
            self._cache[key] = slangtorch.loadModule(
                os.path.join(self.shaders_dir, "raytrace_shader.slang"),
                defines=defines
            )
        return self._cache[key]

# Usage instance
shaders_path = os.path.dirname(__file__)
shader_manager = ShaderManager(shaders_path)
