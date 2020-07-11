

from ...src.nlptools.tasker import tasker  # pylint: disable=relative-beyond-top-level


t = tasker("sentiment")  # loads model
out = t("Deal is good, Happy")  # classifies
print("we won the deal " + str(out))
out = t("Not Responding, will be delayed")
print("we lost deal " + str(out))
out = t(" Nothing can be told about it - like Meeting with Partner and Customer , Follow up with customer on meeting slot")
print(out)
