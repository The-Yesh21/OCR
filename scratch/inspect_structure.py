import inspect
from paddleocr import PPStructureV3

print("PPStructureV3 constructor signature:")
print(inspect.signature(PPStructureV3.__init__))

pps = PPStructureV3()
print("\nPPStructureV3 methods:")
for name, member in inspect.getmembers(pps):
    if not name.startswith("_") and (inspect.ismethod(member) or inspect.isfunction(member)):
        print(f"  {name}")
