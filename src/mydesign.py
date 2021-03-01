import math
import cv2
import numpy as np
import copy
import os, tkinter, tkinter.filedialog, tkinter.messagebox

#HSVテーブル
hsv_table=[[[None for i in range(15)]for j in range(15)]for k in range(30)]
for k in range(15):
    for j in range(15):
        for i in range(30):
            r = k*j
            x = r*math.cos(2.0*math.pi*i/30)
            y = k*15.0
            z = r*math.sin(2.0*math.pi*i/30)
            hsv_table[i][j][k] = [x,y,z]
            


# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
filename = file

img = cv2.imread(filename, 1)

small_img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
resize_img = cv2.resize(small_img, (512,512), interpolation=cv2.INTER_NEAREST)

hsv_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)

for j in hsv_img:
    for i in j:
        i[0] = (i[0]+3)//6
        if 29 < i[0]:
            i[0] = 0

        i[1] = (i[1]*14)//255
        i[2] = (i[2]*14)//255


count_array = [[[0 for i in range(15)]for j in range(15)]for k in range(30)]
change_array = [[[None for i in range(15)]for j in range(15)]for k in range(30)]
for j in hsv_img:
    for i in j:
        count_array[i[0]][i[1]][i[2]] += 1

color_array = []

count = 0
for k in range(15):
    for j in range(15):
        for i in range(30):
            if 0 < count_array[i][j][k]:
                count += 1
                color_array.append([i,j,k])


def hsv_world(h, s, v):
    return hsv_table[h][s][v]
    r = v*s
    x = r*math.cos(2.0*math.pi*h/30)
    y = v*15.0
    z = r*math.sin(2.0*math.pi*h/30)
    return x,y,z

def inv_hsv_world(x, y, z):
    v = round(y/15)
    if 14 < v:
        v = 14
    if v <= 0:
        s = 0
    else:
        s = round(math.sqrt(x*x+z*z)/v)
        if 14 < s:
            s = 14
    t = math.atan2(z, x)
    h = round(t*30/2/math.pi + 30)%30
    return h, s, v

print('変換中です')
n = 0
while True:
    minr = 100
    minxyz = None
    min_color1 = None
    min_color2 = None
    for color1 in color_array:
        for color2 in color_array:
            if color1 != color2:
                tx1, ty1, tz1 = hsv_world(color1[0],color1[1],color1[2])
                tx2, ty2, tz2 = hsv_world(color2[0],color2[1],color2[2])
                tr = (tx1-tx2)*(tx1-tx2) + (ty1-ty2)*(ty1-ty2) + (tz1-tz2)*(tz1-tz2)
                tr *= count_array[color1[0]][color1[1]][color1[2]]
                if minr > tr or minxyz is None:
                    minr = tr
                    minxyz = [color1[0],color1[1],color1[2],color2[0],color2[1],color2[2]]
                    min_color1 = color1
                    min_color2 = color2

    tc1 = count_array[minxyz[0]][minxyz[1]][minxyz[2]]
    tc2 = count_array[minxyz[3]][minxyz[4]][minxyz[5]]
    if True:
        tx1, ty1, tz1 = hsv_world(minxyz[0], minxyz[1], minxyz[2])
        tx2, ty2, tz2 = hsv_world(minxyz[3], minxyz[4], minxyz[5])

        xx = (tx1 * tc1 + tx2 * tc2)/(tc1 + tc2)
        yy = (ty1 * tc1 + ty2 * tc2)/(tc1 + tc2)
        zz = (tz1 * tc1 + tz2 * tc2)/(tc1 + tc2)

        nh, ns, nv = inv_hsv_world(xx, yy, zz)
        count_array[minxyz[0]][minxyz[1]][minxyz[2]] = 0
        count_array[minxyz[3]][minxyz[4]][minxyz[5]] = 0
        count_array[nh][ns][nv] += tc1+tc2
    
    for j in hsv_img:
        for i in j:
            if np.array_equal(i, minxyz[0:3]) or np.array_equal(i, minxyz[3:6]):
                i[0] = nh
                i[1] = ns
                i[2] = nv
    
    color_array.remove(min_color1)
    color_array.remove(min_color2)
    if not [nh,ns,nv] in color_array:
        color_array.append([nh,ns,nv])

    count = len(color_array)
    if count <= 15:
        break
    n += 1
    

    if False:
        hsv_img_t = np.copy(hsv_img)
        for j in hsv_img_t:
            for i in j:
                i[0] *= 6
                i[1] = i[1] * 255//14
                i[2] = i[2] * 255//14
        rgb_img = cv2.cvtColor(hsv_img_t, cv2.COLOR_HSV2BGR)
        resize_rgb_img = cv2.resize(rgb_img, (512,512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('rgb',resize_rgb_img)
        cv2.waitKey(1)


print('')

palette = np.array([[0,0,255],[0,128,255],[0,255,255],[0,255,0],[0,128,0],[255,255,0],[255,128,0],[255,0,0],[255,0,128],[255,0,255],[128,192,255],[0,64,128],[255,255,255],[128,128,128],[0,0,0]])

hsv_palette = []

palette_img = np.copy(hsv_img)

count = 0
for k in range(15):
    for j in range(15):
        for i in range(30):
            if 0 < count_array[i][j][k]:
                hsv_palette.append((i,j,k))
                for l in palette_img:
                    for m in l:
                        if np.array_equal(m, np.array([i,j,k])):
                            m[0] = palette[count][0]
                            m[1] = palette[count][1]
                            m[2] = palette[count][2]
                count += 1
palette_img = cv2.resize(palette_img,(512,512),interpolation=cv2.INTER_NEAREST)
for i in range(32):
    palette_img = cv2.line(palette_img,(i*16,0),(i*16,512),(64,64,64),1)
    palette_img = cv2.line(palette_img,(0,i*16),(512,i*16),(64,64,64),1)
palette_img = cv2.line(palette_img,(254,0),(254,512),(64,64,64),4)
palette_img = cv2.line(palette_img,(0,256),(512,256),(64,64,64),4)
palette_img = cv2.line(palette_img,(384,0),(384,512),(64,64,64),2)
palette_img = cv2.line(palette_img,(0,384),(512,384),(64,64,64),2)
palette_img = cv2.line(palette_img,(128,0),(128,512),(64,64,64),2)
palette_img = cv2.line(palette_img,(0,128),(512,128),(64,64,64),2)
cv2.imshow("default palette", palette_img)
cv2.imwrite('output_palette.png',palette_img)
color_name = ['(赤)　　　','(オレンジ)','(黄)　　　','(黄緑)　　','(緑)　　　','(青緑)　　','(水色)　　','(青)　　　','(紫)　　　','(ピンク)　','(肌色)　　','(茶色)　　','(白)　　　','(灰色)　　','(黒)　　　']
print("                    色相  彩度  明度")
for i in range(15):
    print("{:4}番目{}  {:2}    {:2}    {:2}".format(i+1, color_name[i], hsv_palette[i][0]+1, hsv_palette[i][1]+1, hsv_palette[i][2]+1))

print('')
print('デフォルトのパレットで打ち込んだ後に')
print('各色の色相、彩度、明度を調整してください')
print('右に行くほど大きな値です')
print('色相: 1～30')
print('彩度: 1～15')
print('明度: 1～15')

for j in hsv_img:
    for i in j:
        i[0] *= 6
        i[1] = i[1] * 255//14
        i[2] = i[2] * 255//14
rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
resize_rgb_img = cv2.resize(rgb_img, (512,512), interpolation=cv2.INTER_NEAREST)
cv2.imshow('output',resize_rgb_img)
cv2.imwrite('output.png',resize_rgb_img)

cv2.waitKey(0)