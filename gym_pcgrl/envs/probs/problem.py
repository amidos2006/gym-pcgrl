from PIL import Image

class Problem:
    def __init__(self):
        self._width = 9
        self._height = 9
        self._prob = None

        self._border_size = 1
        self._border_tile = 1
        self._tile_size=16
        self._graphics = None

    def adjust_param(self, **kwargs):
        raise NotImplementedError('get_graphics is not implemented')

    def get_stats(self, map):
        raise NotImplementedError('get_graphics is not implemented')

    def get_reward(self, new_stats, old_stats):
        raise NotImplementedError('get_graphics is not implemented')

    def get_episode_over(self, new_stats, old_stats):
        raise NotImplementedError('get_graphics is not implemented')

    def get_debug_info(self, new_stats, old_stats):
        raise NotImplementedError('get_debug_info is not implemented')

    def render(self, map):
        full_width = self._width+2*self._border_size
        full_height = self._height+2*self._border_size
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,255))
        for y in range(full_height):
            for x in range(self._border_size):
                lvl_image.paste(self._graphics[str(self._border_tile)], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[str(self._border_tile)], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        for x in range(full_width):
            for y in range(self._border_size):
                lvl_image.paste(self._graphics[str(self._border_tile)], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[str(self._border_tile)], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))
        for y in range(self._height):
            for x in range(self._width):
                lvl_image.paste(self._graphics[str(int(map[y][x]))], ((x+self._border_size)*self._tile_size, (y+self._border_size)*self._tile_size, (x+self._border_size+1)*self._tile_size, (y+self._border_size+1)*self._tile_size))
        return lvl_image
