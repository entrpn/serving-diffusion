import base64
import json

with open('../images/ddlm_2.png', "rb") as image_file:
    base64_image = base64.b64encode(image_file.read())
print(type(base64_image))
appDict = {
  "instances": [{"prompt" : "A woman dressed like the Mexican Holiday Dia de los Muertos", "image" : base64_image.decode("utf-8") }],
  "parameters": {
      "ddim_steps" : 50,
      "scale" : 7.5,
      "n_samples" : 2,
      "n_itter" : 2,
      "strength" : .55,
      "type" : "img2img"
  },
}
app_json = json.dumps(appDict)
print(app_json)
with open("request.json", "w") as text_file:
    text_file.write(app_json)