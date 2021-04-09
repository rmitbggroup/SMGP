echo "fennel with q2 on epinion"
motif=q2
baseline=1
dataset=Epinion2
sample=25000
python3 smgp+.py --triangle /Epinion/q0.txt --truth Epinion/q2.txt --baseline $baseline Epinion/  --motif q2 --sample $sample --iteration 10 


echo "mappr with q2 on epinion"
motif=q2
baseline=2
dataset=Epinion2
sample=25000
python3 smgp+.py --triangle /Epinion/q0.txt --truth Epinion/q2.txt --baseline $baseline --output Epinion/  --motif q2 --sample $sample --iteration 10 


