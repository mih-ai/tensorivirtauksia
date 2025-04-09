import tkinter
from tkinter import PhotoImage, Canvas, Checkbutton, Button, Entry, NW
import numpy as np
import time
import cv2
from PIL import Image, ImageTk
import imutils
import os
import mediapipe as mp # hox the --user flag may be needed when installing...
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# and for AI (remove if not needed...)
from tensorflow import keras

__title__=      "Example code for the course of Machine Learning ... can be applied also for something else by a small innovative touch"
__author__ =    "Manne Hannula"
__copyright__ = "Copyright (C) 2025 Manne Hannula"
__email__ =     "etunimi.sukunimi.eiroskapostiakiitos@oamk.fi"
__license__ =   "MIT" # meinaa että esimerkkiä saapi käyttää vapaasti vaikka omissa pisneksissäkin kun vain muistaa mainita lähteen

#===============================================================================================#
#                                                                                               #
#   Remember, this is a quick example code only ...                                             #
#   ...include both Finnish and English comments...do not worry..                               #
#   ...please take your own copy and...                                                         #
#   ...modify it according to instructions in our course or according to some other need!       #
#                                                                                               #
#===============================================================================================#

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

#luodaan kuva, huomaa geometrian asetus
ikkuna=tkinter.Tk()
ikkuna.title("Machine Learning")
ikkuna.geometry("800x500")

#pienta saatoa...
ikoni=PhotoImage(file=r's.png')
ikkuna.iconphoto(False,ikoni)

#taustakuva siten että mahtuu geometriaan...
taustakuva_t=Image.open(r'how_tensors_flow.png')
#taustakuvaskaalaus...
tausta_leveys_alkup=1502
tausta_korkeus_alkup=610
#ja skaalataan uusi kuBa tämän mukaan
taustakuva_t2=taustakuva_t.resize((tausta_leveys_alkup,tausta_korkeus_alkup))
taustakuva=ImageTk.PhotoImage(taustakuva_t2)
uusigeometria=''.join([str(taustakuva.width()),'x',str(taustakuva.height())])
ikkuna.geometry(uusigeometria)

#kanvaasien luonti...
kanvaasi=Canvas(ikkuna,width=800,height=500)
kanvaasi.pack(fill="both",expand=True)
kanvaasi.create_image(0,0,image=taustakuva,anchor="nw")

#alikanvaasi...
nayttoruutukanvaasi = Canvas(ikkuna,width=320,height=240,bg='#ffffff')
#siihen kuva...
paneeli_image=tkinter.Label(ikkuna) #,image=img)
nayttoruutukanvaasi.create_window(1,1,window=paneeli_image,anchor=NW)
#ja yhdistetaan tama ikkunaan...
nayttoruutukanvaasi_ylataso_ikkuna=kanvaasi.create_window(165,185,anchor="nw",window=nayttoruutukanvaasi)

#kanvaasi kasvoille...
kasvo_ikkuna_leveys=120
nayttoruutukanvaasi_kasvot = Canvas(ikkuna,width=kasvo_ikkuna_leveys,height=160,bg='#ffffff')
#siihen kuva...
paneeli_image_kasvot=tkinter.Label(ikkuna) #,image=img)
nayttoruutukanvaasi_kasvot.create_window(1,1,window=paneeli_image_kasvot,anchor=NW)
#ja yhdistetaan tama ikkunaan...
nayttoruutukanvaasi_ylataso_ikkuna_kasvot=kanvaasi.create_window(510,250,anchor="nw",window=nayttoruutukanvaasi_kasvot)

#kanvaasit silmille...
silma_ikkuna_leveys=400
silma_ikkuna_korkeus=210

#kanvaasi silmälle v...
nayttoruutukanvaasi_silma_v = Canvas(ikkuna,width=silma_ikkuna_leveys,height=silma_ikkuna_korkeus,bg='#ffffff')
#siihen kuva...
paneeli_image_silma_v=tkinter.Label(ikkuna) #,image=img)
nayttoruutukanvaasi_silma_v.create_window(1,1,window=paneeli_image_silma_v,anchor=NW)
#ja yhdistetaan tama ikkunaan...
nayttoruutukanvaasi_ylataso_ikkuna_silma_v=kanvaasi.create_window(680,113,anchor="nw",window=nayttoruutukanvaasi_silma_v)

#kanvaasi silmälle o...
nayttoruutukanvaasi_silma_o = Canvas(ikkuna,width=silma_ikkuna_leveys,height=silma_ikkuna_korkeus,bg='#ffffff')
#siihen kuva...
paneeli_image_silma_o=tkinter.Label(ikkuna) #,image=img)
nayttoruutukanvaasi_silma_o.create_window(1,1,window=paneeli_image_silma_o,anchor=NW)
#ja yhdistetaan tama ikkunaan...
nayttoruutukanvaasi_ylataso_ikkuna_silma_o=kanvaasi.create_window(680,345,anchor="nw",window=nayttoruutukanvaasi_silma_o)

#graafikanvaaseja oikealle...
oikea_graafi_kanvaasi = Canvas(ikkuna,width=180,height=120,bg='#0fffff')
oikea_graafi_ikkuna=kanvaasi.create_window(1230,129,anchor="nw",window=oikea_graafi_kanvaasi)

#Tehdään yksi piirtoalusta
fig = Figure(figsize=(2.1, 1.5))
a=fig.add_subplot(111)
a.tick_params(axis='x',labelsize=5)
a.tick_params(axis='y',labelsize=5)

#Näytetään se heti aluksi...
oikea_graafi_canvas = FigureCanvasTkAgg(fig, master=oikea_graafi_kanvaasi)
oikea_graafi_canvas.draw()
oikea_graafi_canvas.get_tk_widget().pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)

#graafikanvaaseja vasemmalle...
vasen_graafi_kanvaasi = Canvas(ikkuna,width=180,height=120,bg='#0fffff')
vasen_graafi_ikkuna=kanvaasi.create_window(1230,370,anchor="nw",window=vasen_graafi_kanvaasi)

#Tehdään yksi piirtoalusta
fig2 = Figure(figsize=(2.1, 1.5))
a2=fig2.add_subplot(111)
a2.tick_params(axis='x',labelsize=5)
a2.tick_params(axis='y',labelsize=5)

#Näytetään se heti aluksi...
vasen_graafi_canvas = FigureCanvasTkAgg(fig2, master=vasen_graafi_kanvaasi)
vasen_graafi_canvas.draw()
vasen_graafi_canvas.get_tk_widget().pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=1)

valintaruutu_arvo = tkinter.IntVar()
valintaruutu_arvo_silmille=tkinter.IntVar()

#Jos malli on saatavilla (huom! random-painoilla luotu esimerkkimalli tyypillisesti jaossa...)
model = keras.models.load_model("example_model.keras")

global silma_v
global silma_o

def aloita():
    global model
    global cap
    global silma_v
    global silma_o
    oikea_tulosvektori=[]
    vasen_tulosvektori=[]
    aikavektori=[]
    luku=0
    #piirto-asetuksia voi luonnostella erilaisia tarpeen mukaan...
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    thin_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
    cap = cv2.VideoCapture(0)

    print("Kameran perustarkkuus:",cap.get(3),cap.get(4))
    frame_leveys=cap.get(3) #1920
    frame_korkeus=cap.get(4) #1080
    #mahdollisesti säädettävä frame ratea ... jos halutaan tarkempaa kuvaa ja jos koneen nopeus ei riita laitetettava pienemmaksi...
    #cap.set(3, frame_leveys)
    #cap.set(4, frame_korkeus)
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame)

        #mesh annotations...
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:

            if valintaruutu_arvo_silmille.get():
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=thin_drawing_spec,
                    connection_drawing_spec=thin_drawing_spec)

            #Tutkitaan kasvojen osaa:
            kasvot_vasen=np.array(face_landmarks.landmark)[234]
            kasvot_oikea=np.array(face_landmarks.landmark)[454]
            kasvot_yla=np.array(face_landmarks.landmark)[10]
            kasvot_ala=np.array(face_landmarks.landmark)[152]

            kasvot_lisa_x=0
            kasvot_lisa_y=0

            kasvot=frame[int(kasvot_yla.y*frame_korkeus-kasvot_lisa_y):int(kasvot_ala.y*frame_korkeus+kasvot_lisa_y),int(kasvot_vasen.x*frame_leveys-kasvot_lisa_x):int(kasvot_oikea.x*frame_leveys+kasvot_lisa_x)]
            kasvot=imutils.resize(kasvot,width=kasvo_ikkuna_leveys)
            kasvot=cv2.cvtColor(kasvot,cv2.COLOR_BGR2RGB)

            if valintaruutu_arvo.get()==1:
                img_update_kasvot = ImageTk.PhotoImage(Image.fromarray(kasvot))
                paneeli_image_kasvot.configure(image=img_update_kasvot)
                paneeli_image_kasvot.image=img_update_kasvot
                paneeli_image_kasvot.update()

            #Tutktaan lisaa...oikea silma...
            #Tällä tavoin ko. dataframesta saadaan tieto ulos...
            mika_kohta=133
            kohta=np.array(face_landmarks.landmark)[mika_kohta]
            #print(int(kohta.x*640))
            #lisätään tämän tiedon perusteella lisäpiirre kuvaan...
            paksuus=3

            silma_o_vasen=np.array(face_landmarks.landmark)[130]
            silma_o_oikea=np.array(face_landmarks.landmark)[243]
            silma_o_yla=np.array(face_landmarks.landmark)[27]
            silma_o_ala=np.array(face_landmarks.landmark)[23]

            silma_lisa_x=0
            silma_lisa_y=0

            silma_o=frame[int(silma_o_yla.y*frame_korkeus-silma_lisa_y):int(silma_o_ala.y*frame_korkeus+silma_lisa_y),int(silma_o_vasen.x*frame_leveys-silma_lisa_x):int(silma_o_oikea.x*frame_leveys+silma_lisa_x)]
            silma_o=imutils.resize(silma_o,width=silma_ikkuna_leveys)

            if valintaruutu_arvo.get()==1:
                cv2.circle(frame,(int(kohta.x*frame_leveys),int(kohta.y*frame_korkeus)),15,(0,0,255),paksuus)

            if valintaruutu_arvo.get()==1:
                frame_nayta_silma_o=cv2.cvtColor(silma_o,cv2.COLOR_BGR2RGB)
                img_update_silma_o = ImageTk.PhotoImage(Image.fromarray(frame_nayta_silma_o))
                paneeli_image_silma_o.configure(image=img_update_silma_o)
                paneeli_image_silma_o.image=img_update_silma_o
                paneeli_image_silma_o.update()

            #Tutktaan lisaa...oikea silma...

            silma_v_vasen=np.array(face_landmarks.landmark)[463]
            silma_v_oikea=np.array(face_landmarks.landmark)[359]
            silma_v_yla=np.array(face_landmarks.landmark)[257]
            silma_v_ala=np.array(face_landmarks.landmark)[253]

            silma_lisa_x=0
            silma_lisa_y=0

            silma_v=frame[int(silma_v_yla.y*frame_korkeus-silma_lisa_y):int(silma_v_ala.y*frame_korkeus+silma_lisa_y),int(silma_v_vasen.x*frame_leveys-silma_lisa_x):int(silma_v_oikea.x*frame_leveys+silma_lisa_x)]
            silma_v=imutils.resize(silma_v,width=silma_ikkuna_leveys)

            if valintaruutu_arvo.get()==1:
                frame_nayta_silma_v=cv2.cvtColor(silma_v,cv2.COLOR_BGR2RGB)
                img_update_silma_v = ImageTk.PhotoImage(Image.fromarray(frame_nayta_silma_v))
                paneeli_image_silma_v.configure(image=img_update_silma_v)
                paneeli_image_silma_v.image=img_update_silma_v
                paneeli_image_silma_v.update()

            #Tällä tavoin ko. dataframesta saadaan tieto ulos...
            mika_kohta=362
            kohta=np.array(face_landmarks.landmark)[mika_kohta]
            #print(int(kohta.x*640))
            #lisätään tämän tiedon perusteella lisäpiirre kuvaan...
            if valintaruutu_arvo.get()==1:
                paksuus=3
                cv2.circle(frame,(int(kohta.x*frame_leveys),int(kohta.y*frame_korkeus)),15,(0,0,255),paksuus)


            #=====================================================================================================
            #
            #       A  N  A  L  Y  S  I  S
            #
            #=====================================================================================================
            #
            # There is the room for the analysis...after training the artificial intelligence implement it here
            #... to make the analysis...
            # ...to load the model use command like model = keras.models.load_model('the_model_I_have_trained.keras')
            #

            luku=luku+1
            aikavektori.append(luku)

            #simulated here...replace with your own code...
            right_eye_some_result=100*np.random.rand()

            #muunnetaan matriisi kuvaksi...skaalaus...matriisiksi...mallin lapi...
            kuva=Image.fromarray(silma_v, 'RGB')
            kuva_vakiokoko_v=kuva.resize((400,200))
            kuva_vakiokoko = np.array(kuva_vakiokoko_v)
            silma_o_syote=np.expand_dims(kuva_vakiokoko,axis=0)
            right_eye_some_result=model(silma_o_syote)

            #this way you can investigat the result in detail:
            #print(np.array(right_eye_some_result)[0])

            #right_eye_some_result=np.array(right_eye_some_result)
            temp=np.array(right_eye_some_result)[0]
            oikea_tulosvektori.append(temp)

            #simulated here...replace with your own code...
            left_eye_some_result=1*np.random.rand()

            #muunnetaan matriisi kuvaksi...skaalaus...matriisiksi...mallin lapi...
            kuva=Image.fromarray(silma_o, 'RGB')
            kuva_vakiokoko_v=kuva.resize((400,200))
            kuva_vakiokoko = np.array(kuva_vakiokoko_v)
            silma_o_syote=np.expand_dims(kuva_vakiokoko,axis=0)
            left_eye_some_result=model(silma_o_syote)

            temp=np.array(left_eye_some_result)[0]
            vasen_tulosvektori.append(temp)

            #vasen_tulosvektori.append(left_eye_some_result)

            #
            #
            #=====================================================================================================

            #tehdään vielä graafit:
            if valintaruutu_arvo.get()==1:
                a.cla()
                a.plot(aikavektori,oikea_tulosvektori,'r')
                a.set_xlim([np.max([0,len(aikavektori)-20]),len(aikavektori)])
                a.set_ylim([-0.1,1.1])
                oikea_graafi_canvas.draw()
                oikea_graafi_canvas.get_tk_widget().update()

                a2.cla()
                a2.plot(aikavektori,vasen_tulosvektori,'r')
                a2.set_xlim([np.max([0,len(aikavektori)-20]),len(aikavektori)])
                a2.set_ylim([-0.1,1.1])
                vasen_graafi_canvas.draw()
                vasen_graafi_canvas.get_tk_widget().update()

        frame_pikku=imutils.resize(frame,width=320)
        frame_nayta=cv2.cvtColor(frame_pikku,cv2.COLOR_BGR2RGB)
        img_update = ImageTk.PhotoImage(Image.fromarray(frame_nayta))
        paneeli_image.configure(image=img_update)
        paneeli_image.image=img_update
        paneeli_image.update()

        if cv2.waitKey(5) & 0xFF == 27:
          break

def lopeta():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped!")

#============================================================================================================
#
#  C R E A T E   Y O U R   O W N   D A T A   S E T   U S I N G  S I M P L E  F O L D E R   S T U C T U R E
#
#============================================================================================================
#
# Note - the idea and logic about the folders is described in the lessons...

def talleta():
    global silma_v
    global silma_o
    print("Tallennetaan kuvat, seka opetus, validointi etta testidata...")
    #annetaan kuvalle nimi...
    #...ja jos kansiota ei ole, luodaan...
    if os.path.isdir(''.join([kansionimi.get(),'_left/'])) & os.path.isdir(''.join([kansionimi.get(),''])):
        kuvanimi_left=''.join([kansionimi.get(),'_left/',str(int(time.time())),'.png'])
        kuvanimi_right=''.join([kansionimi.get(),'/',str(int(time.time())),'.png'])
    else:
        print("luodaan uusi kansio...")
        try:
            os.mkdir(''.join([kansionimi.get(),'_left/']))
        except:
            print("jokin virhe")
        try:
            os.mkdir(''.join([kansionimi.get(),'']))
        except:
            print("jokin virhe")

        kuvanimi_left=''.join([kansionimi.get(),'_left/',str(int(time.time())),'.png'])
        kuvanimi_right=''.join([kansionimi.get(),'/',str(int(time.time())),'.png'])

    #Tallennukset:
    print("...nimilla: ",kuvanimi_left,kuvanimi_right)
    # HOX vasen ja oikea peilautuu, huomaa!
    #cv2.imwrite(kuvanimi_left, silma_v)
    #cv2.imwrite(kuvanimi_right, silma_o)
    cv2.imwrite(kuvanimi_left, silma_o)
    cv2.imwrite(kuvanimi_right, silma_v)

painikkeiston_sijainti_x=280
painikkeiston_sijainti_y=450
painikkeiston_valistus=30

kansionimi=tkinter.StringVar()
kansionimi.set('silma')

valintaruutu=Checkbutton(ikkuna, text='Show more...',variable=valintaruutu_arvo)
valintaruutu_ikkuna=kanvaasi.create_window(230,150,anchor="nw",window=valintaruutu)

valintaruutu_silmille=Checkbutton(ikkuna, text='...and more...',variable=valintaruutu_arvo_silmille)
valintaruutu_ikkuna_silmille=kanvaasi.create_window(330,150,anchor="nw",window=valintaruutu_silmille)

kansioruutu=Entry(ikkuna,textvariable=kansionimi)
kansioruutu_ikkuna=kanvaasi.create_window(940,70,anchor="nw",window=kansioruutu)

painike1=Button(ikkuna,text="Start video",command=aloita)
painike1_ikkuna=kanvaasi.create_window(painikkeiston_sijainti_x-painikkeiston_valistus,painikkeiston_sijainti_y,anchor="nw",window=painike1)

painike2=Button(ikkuna,text="Stop video",command=lopeta)
painike2_ikkuna=kanvaasi.create_window(painikkeiston_sijainti_x+2*painikkeiston_valistus,painikkeiston_sijainti_y,anchor="nw",window=painike2)

painike_talleta=Button(ikkuna,text="Save the image below to a folder:",command=talleta)
painike_talleta_ikkuna=kanvaasi.create_window(740,70,anchor="nw",window=painike_talleta)

ikkuna.mainloop()

#Täällä voisi olla vielä jotakin muutakin koodia...
#
#
#...mutta juuri nyt ei ole!
