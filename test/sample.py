from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
out = predictor.predict(
    sentence="Megan Maples hasn't got back to me yet on Martin it. Morgan Stanley gave us the 25th or the 26. I sent her a note. I think I copied you want it? Yesterday? I just needed a day at a time back from her, which is works best for Trevor. And I'm good there and then, you know, and maybe this is an assigned. But if I start to get some meetings, Book like, in particular Cro Nos and Seismic. You a meeting with Michael tentatively, Michael McDonagh on Thursday. But if we start to get some things going up here in the Northeast, maybe Brent that I can It's going to get you out here and, you know, you know, keep you busy for, like, a week meeting with clients, and you know so but that's something that, you know, maybe is more March. March. Focus."
)
# out['words']
print(out['tags'])
out_dict = {}
for i, j in zip(out['words'], out['tags']):
    if j == "U-ORG" or j == "B-ORG":
        out_dict[i] = "Organization"
    elif j == "U-LOC":
        out_dict[i] = "Location"
    elif j == "U-PER" or j == "B-PER":
        out_dict[i] = "Person"
    elif j == "O":
        pass
    #out_dict[i] = j

print(out_dict)
