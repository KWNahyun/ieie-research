import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/manuscript_deps'))
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
p=Path(__file__).resolve().parent/'cnn_architecture.pdf'
c=canvas.Canvas(str(p),pagesize=(300,175))
def box(x,y,w,h,title,sub='',fill='#eef3f8'):
 c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor('#53677b'));c.roundRect(x,y,w,h,3,fill=1,stroke=1)
 c.setFillColor(HexColor('#172b3f'));c.setFont('Helvetica-Bold',8);c.drawCentredString(x+w/2,y+h/2+(3 if sub else -2),title)
 if sub:c.setFont('Helvetica',7);c.drawCentredString(x+w/2,y+h/2-8,sub)
def arrow(x,y,u,v):
 c.setStrokeColor(HexColor('#53677b'));c.line(x,y,u,v);c.line(u,v,u-3,v+2);c.line(u,v,u-3,v-2)
box(5,115,50,30,'S','3 channels');box(80,115,95,30,'CNN encoder','16 / 32 / 64 / 128');box(200,115,95,30,'Decoder + head','linear output')
arrow(55,130,80,130);arrow(175,130,200,130)
box(5,60,50,30,'P','H / V / n2i');box(80,60,95,30,'1 x 1 projection','zero initialization')
arrow(55,75,80,75);c.line(127,90,127,115)
box(5,5,50,30,'Q','5 dimensions');box(80,5,95,30,'MLP / FiLM','5 -> 32 -> 256')
arrow(55,20,80,20);c.line(175,20,185,20);c.line(185,20,185,105);c.line(185,105,158,105);c.line(158,105,158,115)
c.setFont('Helvetica',7);c.drawString(199,93,'Skip connections');c.drawString(199,81,'Full target fine-tuning')
c.setDash(3,2);c.line(90,151,90,161);c.line(90,161,244,161);c.line(244,161,244,145)
c.setDash();c.save();print(p)
