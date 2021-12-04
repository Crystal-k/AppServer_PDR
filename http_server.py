import json
import re
import numpy as np
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import utils as pdr


termLoc = [[0,0]]
x = [0]
y = [0]
steps = 0
gravity_g = [[]]
linear_g = [[]]
rotation_g = [[]]


class AndroidHTTPHandler(BaseHTTPRequestHandler):
    # == 通信处理 ============================
    def do_POST(self):
        print("收到post")

        req_datas = self.rfile.read(int(self.headers['Content-Length'])).decode()  # 获取post数据
        data = json.loads(req_datas)  # 转化为python可处理格式

        # print("data",data)
        flag = data['flag']
        print("flag",flag)
        if flag:
            response = self.processPDR(data, pdr)
        else:
            print("reset")
            response = {'status': 'true',
                        'TermLoc': termLoc[-1]}
            self.reset()


        f = open('./headersistory.txt', mode='a', encoding='utf-8')
        f.write('{}\n'.format(response))
        f.close()

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        # for ajax
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")

        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    # # == 定位请求处理 ============================
    # 处理PDR定位请求
    def processPDR(self, data, pdr):
        # print(data)
        # 解析JSON数据
        global x,y,steps,gravity_g,linear_g,rotation_g,termLoc

        grav_data = data['GravData']
        linear_acc_data = data["LinearAccData"]
        rotation_data = data['RotationData']

        if len(grav_data) != 0 and len(linear_acc_data) != 0 and len(rotation_data) != 0:
            arr_grav = re.split('[,]', grav_data)
            arr_grav = list(map(eval, arr_grav))
            len_gav = arr_grav.__len__()

            arr_linear = re.split('[,]', linear_acc_data)
            arr_linear = list(map(eval, arr_linear))
            len_linear = arr_linear.__len__()

            arr_rotation = re.split('[,]', rotation_data)
            arr_rotation = list(map(eval, arr_rotation))
            len_rotation = arr_rotation.__len__()

            if len_gav <= 3 or len_linear <= 3 or len_rotation <= 4 \
                    or (len_gav/3 != len_linear/3): #or (len_gav/3 != len_rotation/4):
                print("data error")
                print("gravity", len_gav / 3)
                print("linear acc", len_linear / 3)
                print("rotation", len_rotation / 4)
                print("===============")
                # print("gravity", len_gav / 3)
                # print("linear acc", len_linear/ 3)
                # print("rotation", len_rotation/ 4)
                # print("===============")

                response = {'status': 'true',
                            'TermLoc': termLoc[-1]}  # 到时候替换即可

                return response

            arr_grav = np.array(arr_grav)
            arr_linear = np.array(arr_linear)
            arr_rotation = np.array(arr_rotation)

            print("gravity", len_gav / 3)
            print("linear acc", len_linear/ 3)
            print("rotation", len_rotation/ 4)
            print("===============")

            gravity = arr_grav.reshape(int(arr_grav.shape[0] / 3), 3)
            linear = arr_linear.reshape(int(arr_linear.shape[0] / 3), 3)
            rotation = arr_rotation.reshape(int(arr_rotation.shape[0] / 4), 4)

            if np.array_equal(gravity_g, [[]]):
                gravity_g = gravity
                linear_g = linear
                rotation_g = rotation
            else:
                gravity_g = np.vstack((gravity_g, gravity))
                linear_g = np.vstack((linear_g, linear))
                rotation_g = np.vstack((rotation_g, rotation))

            pdr = pdr.Model(linear_g, gravity_g, rotation_g)
            x, y, steps = pdr.show_trace(frequency=100, walkType='normal', initPosition=(0, 0), offset=0)
            loc = [round(x[steps],2), round(y[steps],2)]
            termLoc.append(loc)
            print(loc)

        response = {'status': 'true', 'TermLoc': loc}

        return response

    def reset(self):
        global x, y, steps, gravity_g, linear_g, rotation_g,termLoc
        termLoc = [[0, 0]]
        x = [0]
        y = [0]
        steps = 0
        gravity_g = [[]]
        linear_g = [[]]
        rotation_g = [[]]



if __name__ == '__main__':
    # 启动HTTP服务，此处需使用电脑打开热点，此时IP就是192.168.137.1，然后使用移动设备连接电脑热点，即可实现通信
    server = HTTPServer(('192.168.137.1', 8001), AndroidHTTPHandler)
    # server = HTTPServer(('192.168.0.186', 8001), AndroidHTTPHandler)   # 研究院服务器公网IP

    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()

