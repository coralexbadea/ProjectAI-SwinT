from .models.fire_forest_model import LoadModel
import gc 

class FireForestService():
   def __init__(self) -> None:
      self.loadModel = LoadModel()
      self.loadModel.createModel()
      # pass

   def generate(self, image):
      # loadModel = LoadModel()
      
      result = self.loadModel.predict(image)
      # del loadModel
      gc.collect()
      return result