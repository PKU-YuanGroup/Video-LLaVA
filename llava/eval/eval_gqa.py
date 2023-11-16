# Evaluation code for GQA.
# Computes a suite of metrics such as accuracy, consistency, plausibility and scores per question type and length.
# Visit https://gqadataset.org/ for all information about the dataset, including examples, visualizations, paper and slides.
#
#
# Metrics:
# - Accuracy: Standard accuracy, computed over the balanced version of the dataset, which is more robust against
#             cheating by making educated guesses. For each question-answer pair (q,a), we give 1 point if the
#             predicted answer p matches a and 0 otherwise, and average over all questions in the dataset.
#
# - Consistency: A metric for the level of model's consistency across different questions. For each question-answer
#                pair (q,a), we define a set Eq={q1, q2, ..., qn} of entailed questions, the answers to which can
#                be unambiguously inferred given (q,a).
#                Denote Q the set of all questions the model answered correctly. For each question q in Q, we
#                measure the model's accuracy over the entailed questions Eq to get the score sq and finally
#                average these results across all questions in Q.
#
# - Validity: Measures whether the model gives a "valid" answer - one that can theoretically be an answer
#             to the question (e.g. a color to a color question, yes/no to a binary question etc.).
#             We provide a set of valid answers to each questions over the final answer vocabulary, in
#             the choices file, and use it to compute average validity across the dataset.
#
# - Plausibility: Measures whether the model answers are plausible, e.g. one that make sense in the real world,
#                 e.g. not answering "purple" to a question about apple color (unless it's really purple).
#                 We provide a set of all plausible answers to each questions, computed by looking at all
#                 attributes and relations hold for various objects throughout the whole dataset scene graphs,
#                 and use it to compute average model plausibility across the data.
#
# - Grounding: Only for attention models. Measures whether the model looks at the relevant regions in the
#              image when answering a question. Each question in the dataset is annotated with the visual regions
#              they refer to, which are then used to compute the level to which the model has a correct visual attention,
#              which will allow to identify whether it really answers based on the image of by language-based guesses.
#              Supports both spatial features and object-based features.
#
# - Distribution: Measures the overall match between the true answer distribution for different questions,
#                 vs the overall distribution predicted by the model through its answers for all the data.
#                 We use chi-square statistic to measure the degree of similarity between the distributions,
#                 giving indication to the level of overall world-knowledge of the model
#
# - Accuracy per type: accuracy per question structural types (logic, compare, choose), and semantic type
#                      (questions about attributes, relations, categories, objects or the whole scene).
#
# - Accuracy for length: accuracy as a function of the question length, in terms of (1) words number, and semantic
#                        complexity - number of reasoning steps.
#
# We may support additional metrics (e.g. coverage) in the future.
#
#
# Files format:
# - predictions file format: JSON array: [{"questionId": str, "prediction": str}]
# - attentions file format: JSON array:
#   Spatial attention: [{"questionId": str, "attention": [mapSize x mapSize: float] }].
#   Object-based attention:[{"questionId": str, "attention": [[x0, y0, x1, y1, float] x #regions] }]. 0 < x,y < 1.
# - questions and choices files are provided as part of the dataset.
#   see https://gqadataset.org/download.html for information about their format.
#
#
# If you have any questions or comments, please feel free to send an email,
# at dorarad@cs.stanford.edu. We hope you'll enjoy using the GQA dataset! :)
#
#

from collections import defaultdict
from tqdm import tqdm
import argparse
import os.path
import glob
import json
import math

##### Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--tier', default="val", type=str, help="Tier, e.g. train, val")
parser.add_argument('--scenes', default="{tier}_sceneGraphs.json", type=str, help="Scene graphs file name format.")
parser.add_argument('--questions', default="{tier}_all_questions.json", type=str, help="Questions file name format.")
parser.add_argument('--choices', default="{tier}_choices.json", type=str, help="Choices file name format.")
parser.add_argument('--predictions', default="{tier}_predictions.json", type=str, help="Answers file name format.")
parser.add_argument('--attentions', default="{tier}_attentions.json", type=str, help="Attentions file name format.")
parser.add_argument('--consistency', action="store_true",
                    help="True to compute consistency score (Need to provide answers to questions in val_all_questions.json).")
parser.add_argument('--grounding', action="store_true",
                    help="True to compute grounding score (If model uses attention).")
parser.add_argument('--objectFeatures', action="store_true",
                    help="True for object-based attention (False for spatial).")
parser.add_argument('--mapSize', default=7, type=int,
                    help="Optional, only to get attention score. Images features map size, mapSize * mapSize")
args = parser.parse_args()

print(
    "Please make sure to use our provided visual features as gqadataset.org for better comparability. We provide both spatial and object-based features trained on GQA train set.")
print(
    "In particular please avoid using features from https://github.com/peteanderson80/bottom-up-attention since they were trained on images contained in the GQA validation set and thus may give false scores improvement.\n")

if not args.consistency:
    print("Please consider using --consistency to compute consistency scores for entailed questions.")
    print("If you do so, please provide answers to all questions in val_all_questions.json.\n")

if not args.grounding:
    print("Please consider using --grounding to compute attention scores.")
    print("If you do so, please provide attention maps through --attentions.\n")


##### Files Loading
##########################################################################################

def loadFile(name):
    # load standard json file
    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)
    # load file chunks if too big
    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir=name.split(".")[0], ext=name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data


# Load scene graphs
print("Loading scene graphs...")
try:
    scenes = loadFile(args.scenes.format(tier=args.tier))
except:
    print('Failed to load scene graphs -- cannot evaluate grounding')
    scenes = None  # for testdev

# Load questions
print("Loading questions...")
questions = loadFile(args.questions)

# Load choices
print("Loading choices...")
try:
    choices = loadFile(args.choices.format(tier=args.tier))
except:
    print('Failed to load choices -- cannot evaluate validity or plausibility')
    choices = None  # for testdev

# Load predictions and turn them into a dictionary
print("Loading predictions...")
predictions = loadFile(args.predictions.format(tier=args.tier))
predictions = {p["questionId"]: p["prediction"] for p in predictions}

# Make sure all question have predictions
for qid in questions:
    if (qid not in predictions) and (args.consistency or questions[qid]["isBalanced"]):
        print("no prediction for question {}. Please add prediction for all questions.".format(qid))
        raise Exception("missing predictions")

# Load attentions and turn them into a dictionary
attentions = None
if args.grounding:
    with open(args.attentions.format(tier=args.tier)) as attentionsFile:
        attentions = json.load(attentionsFile)
        attentions = {a["questionId"]: a["attention"] for a in attentions}


##### Scores data structures initialization
##########################################################################################

# book to float
def toScore(b):
    return float(1 if b else 0)


# Compute average of a list
def avg(l):
    if len(l) == 0:
        return 0
    return float(sum(l)) / len(l)


def wavg(l, w):
    if sum(w) == 0:
        return None
    return float(sum(l[i] * w[i] for i in range(len(l)))) / sum(w)


# Initialize data structure to track all metrics: e.g. accuracy, validity and plausibility, as well as
# accuracy per question type, length and number of reasoning steps.
scores = {
    "accuracy": [],  # list of accuracies per question (1 if correct else 0). Will be averaged ultimately.
    "binary": [],  # list of accuracies per a binary question (1 if correct else 0). Will be averaged ultimately.
    "open": [],  # list of accuracies per an open question (1 if correct else 0). Will be averaged ultimately.
    "validity": [],  # list of validity per question (1 if valid else 0).
    "plausibility": [],  # list of plausibility per question (1 if plausible else 0).
    "consistency": [],  # list of consistency scores for entailed questions.
    "accuracyPerStructuralType": defaultdict(list),
    # list of question accuracies for each structural type (e.g. compare, logic questions).
    "accuracyPerSemanticType": defaultdict(list),
    # list of question accuracies for each semantic type (e.g. questions about an object, an attribute, a relation).
    "accuracyPerLength": defaultdict(list),  # list of question accuracies per question's word number.
    "accuracyPerSteps": defaultdict(list),
    # list of question accuracies per question's reasoning length (steps number).
    "grounding": []  # list of grounding scores for each question.
}

# Initialize golden and predicted histograms per each question group. Used to compute the distribution metric.
dist = {
    "gold": defaultdict(lambda: defaultdict(int)),
    "predicted": defaultdict(lambda: defaultdict(int))
}


##### Question lengths - words numbers and reasoning steps number
##########################################################################################

# Compute question length (words number)
def getWordsNum(question):
    return len(question["question"].split())


# Compute number of reasoning steps (excluding the final "querying" step which doesn't increase effective reasoning length)
def getStepsNum(question):
    return len([c for c in question["semantic"] if not (any([o in "{}: {}".format(c["operation"], c["argument"])
                                                             for o in ["exist", "query: name", "choose name"]]))])


##### Functions for question annotations
##########################################################################################

# Utility function for converting question annotations string keys to slices
def toSlice(strSlice):
    sliceLims = (int(n) for n in strSlice.split(':'))
    return apply(slice, sliceLims)


# Utility function for converting question annotations string keys to indexes list:
# "1" => [0]
# "1:3" => [1, 2]
# "4:9:2" => [4, 6, 8]
def intsFromSlice(strSlice):
    slice_obj = get_slice_obj(slicearg)
    return (range(slice_obj.start or 0, slice_obj.stop or -1, slice_obj.step or 1))


##### Functions for validity and plausibility
##########################################################################################

def belongs(element, group, question):
    # normalization ()
    if "Common" in question["types"]["detailed"]:
        group = ["color", "material", "shape"]

    return element in group


##### Functions for consistency scores (for entailed questions ("inferred"))
##########################################################################################

def updateConsistency(questionId, question, questions):
    inferredQuestions = [eid for eid in question["entailed"] if eid != questionId]

    if correct and len(inferredQuestions) > 0:

        cosnsitencyScores = []
        for eid in inferredQuestions:
            gold = questions[eid]["answer"]
            predicted = predictions[eid]
            score = toScore(predicted == gold)
            cosnsitencyScores.append(score)

        scores["consistency"].append(avg(cosnsitencyScores))


##### Functions for grounding score (optional, only for attention models)
##########################################################################################

# Utility functions for working with bounding boxes.
# c = (x0, y0, x1, y1), r = (r0, r1)

def yrange(c):
    return (c[1], c[3])


def xrange(c):
    return (c[0], c[2])


def length(r):
    if r is None:
        return 0
    return float(r[1] - r[0])


def size(c):
    return length(xrange(c)) * length(yrange(c))


def intersection(r1, r2):
    ir = (max(r1[0], r2[0]), min(r1[1], r2[1]))
    if ir[1] > ir[0]:
        return ir
    return None


def intersectionSize(c1, c2):
    return length(intersection(xrange(c1), xrange(c2))) * length(intersection(yrange(c1), yrange(c2)))


def intersectionRate(c1, c2):
    return float(intersectionSize(c1, c2)) / size(c1)


# Get spatial cell
def getCell(i, j):
    edge = float(1) / args.mapSize
    return (edge * i, edge * j, edge * (i + 1), edge * (j + 1))


# Get bounding box of objectId in sceneGraph
def getRegion(sceneGraph, objectId):
    obj = sceneGraph["objects"][objectId]
    x0 = float(obj["x"]) / sceneGraph["width"]
    y0 = float(obj["y"]) / sceneGraph["height"]
    x1 = float(obj["x"] + obj["w"]) / sceneGraph["width"]
    y1 = float(obj["y"] + obj["h"]) / sceneGraph["height"]
    return (x0, y0, x1, y1)


# Compute grounding score. Computer amount of attention (probability) given to each of the regions
# the question and answers refer to.
def computeGroundingScore(question, sceneGraph, attentionMap):
    ## prepare gold regions
    regions = []
    # add question regions
    regions += [getRegion(sceneGraph, pointer) for pointer in question["annotations"]["question"].values()]
    # add answer regions
    regions += [getRegion(sceneGraph, pointer) for pointer in question["annotations"]["fullAnswer"].values()]
    # add all the image if the question refers to the whole scene
    if any(("scene" in c) for c in question["semantic"]):
        regions.append((0, 0, 1, 1))

    # prepare attention map
    if args.objectFeatures:
        cells = [((x0, y0, x1, y1), attention) for x0, y0, x1, y1, attention in cells]
    else:
        cells = [(getCell(i, j), attentionMap[i][j]) for i in range(args.mapSize) for j in range(args.mapSize)]

    # compare attention map to gold regions
    scores = []
    for region in regions:
        for cell, attention in cells:
            scores.append(attention * intersectionRate(cell, region))
    return sum(scores)


##### Functions for distribution score
##########################################################################################

# Compute chi square statistic of gold distribution vs predicted distribution,
# averaged over all question groups
def chiSquare(goldDist, predictedDist):
    sumScore, sumOverall = 0, 0

    for group in goldDist:
        score, overall = 0, 0

        for ans in goldDist[group]:
            e = goldDist[group][ans]
            o = predictedDist[group].get(ans, 0)
            score += ((float(o - e) ** 2) / e)
            overall += goldDist[group][ans]

        sumScore += score * overall
        sumOverall += overall

    avgScore = float(sumScore) / sumOverall

    return avgScore


##### Main score computation
##########################################################################################

# Loop over the questions and compute mterics
for qid, question in tqdm(questions.items()):

    # Compute scores over the balanced dataset (more robust against cheating by making educated guesses)
    if question["isBalanced"]:
        gold = question["answer"]
        predicted = predictions[qid]

        correct = (predicted == gold)
        score = toScore(correct)

        wordsNum = getWordsNum(question)
        stepsNum = getStepsNum(question)

        # Update accuracy
        scores["accuracy"].append(score)
        scores["accuracyPerLength"][wordsNum].append(score)
        scores["accuracyPerSteps"][stepsNum].append(score)
        scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
        scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
        answerType = "open" if question["types"]["structural"] == "query" else "binary"
        scores[answerType].append(score)

        # Update validity score
        valid = (
            belongs(predicted, choices[qid]["valid"], question) if choices
            else False)
        scores["validity"].append(toScore(valid))

        # Update plausibility score
        plausible = (
            belongs(predicted, choices[qid]["plausible"], question) if choices
            else False)
        scores["plausibility"].append(toScore(plausible))

        # Optionally compute grounding (attention) score
        if attentions is not None:
            groundingScore = computeGroundingScore(question, scenes[question["imageId"]], attentions[qid])
            if groundingScore is not None:
                scores["grounding"].append(groundingScore)

        # Update histograms for gold and predicted answers
        globalGroup = question["groups"]["global"]
        if globalGroup is not None:
            dist["gold"][globalGroup][gold] += 1
            dist["predicted"][globalGroup][predicted] += 1

        if args.consistency:
            # Compute consistency (for entailed questions)
            updateConsistency(qid, question, questions)

# Compute distribution score
scores["distribution"] = chiSquare(dist["gold"], dist["predicted"]) / 100

# Average scores over all questions (in the balanced dataset) and print scores

metrics = [
    "binary",
    "open",
    "accuracy",
    "consistency",
    "validity",
    "plausibility",
    "grounding",
    "distribution"
]

detailedMetrics = [
    ("accuracyPerStructuralType", "Accuracy / structural type"),
    ("accuracyPerSemanticType", "Accuracy / semantic type"),
    ("accuracyPerSteps", "Accuracy / steps number"),
    ("accuracyPerLength", "Accuracy / words number")
]

subMetrics = {
    "attr": "attribute",
    "cat": "category",
    "global": "scene",
    "obj": "object",
    "rel": "relation"
}
# average
for k in metrics:
    if isinstance(scores[k], list):
        scores[k] = avg(scores[k]) * 100

for k, _ in detailedMetrics:
    for t in scores[k]:
        scores[k][t] = avg(scores[k][t]) * 100, len(scores[k][t])

# print
print("")
for m in metrics:
    # skip grounding and consistency scores if not requested
    if m == "grounding" and not args.grounding:
        continue
    if m == "consistency" and not args.consistency:
        continue

    # print score
    print("{title}: {score:.2f}{suffix}".format(title=m.capitalize(), score=scores[m],
                                                suffix=" (lower is better)" if m == "distribution" else "%"))

for m, mPrintName in detailedMetrics:
    print("")
    # print metric title
    print("{}:".format(mPrintName))

    for t in sorted(list(scores[m].keys())):
        # set sub-metric title
        tName = t
        if isinstance(scores[k], list):
            tName = subMetrics.get(t, t).capitalize()

        # print score
        print("  {title}: {score:.2f}{suffix} ({amount} questions)".format(title=tName,
                                                                           score=scores[m][t][0], suffix="%",
                                                                           amount=scores[m][t][1]))