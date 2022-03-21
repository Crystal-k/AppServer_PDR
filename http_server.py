import json
import re
import numpy as np
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import utils as pdr

termLoc = [[0, 0]]
x = [0]
y = [0]
steps = 0
gravity_g = [[]]
linear_g = [[]]
rotation_g = [[]]
orientation_g = [[]]

class AndroidHTTPHandler(BaseHTTPRequestHandler):
    # == 通信处理 ============================
    def do_POST(self):
        print("收到post")

        req_datas = self.rfile.read(int(self.headers['Content-Length'])).decode()  # 获取post数据
        data = json.loads(req_datas)  # 转化为python可处理格式

        # print("data",data)
        # flag = data['flag']
        # print("flag",flag)
        # if flag:
        response = self.processPDR(data, pdr)
        # else:
        #     print("reset")
        #     print("===============")
        #     response = {'status': 'true',
        #                 'TermLoc': termLoc[-1]}
        #     self.reset()

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
        global x, y, steps, linear_g, orientation_g, termLoc

        # linear_acc_data = data["LinearAccData"]  # ours
        linear_acc_data = data["linearAccData"]
        # orientation_data = data["OrientData"]  # ours
        orientation_data = data["deviceData"]

        if len(linear_acc_data) != 0 and len(orientation_data) != 0:
            arr_linear = re.split('[,]', linear_acc_data)
            arr_linear = list(map(eval, arr_linear))
            arr_linear = np.array(arr_linear)

            arr_ori = re.split('[,]', orientation_data)
            arr_ori = list(map(eval, arr_ori))
            arr_ori = np.array(arr_ori)

            if arr_linear.shape[0] <= 3 or arr_ori.shape[0] <= 3 \
                    or (arr_linear.shape != arr_ori.shape):
                print("data shape error")

                print("linear acc's shape: [", arr_linear.shape[0]/3, ", 3]")
                print("orientation's shape: [", arr_ori.shape[0]/3, ", 3]")
                print("===============")

                # min_len = min(len_linear, len_ori)
                # print("规整后长度：", min_len/3)
                #
                # arr_linear = arr_linear[-min_len:]
                # arr_ori = arr_ori[-min_len:]

                response = {'status': 'true',
                            'TermLoc': termLoc[-1]}  # 到时候替换即可

                return response


            linear = arr_linear.reshape(int(arr_linear.shape[0] / 3), 3)
            orientation = arr_ori.reshape(int(arr_ori.shape[0] / 3), 3)

            print("linear acc's shape: [", arr_linear.shape[0], ", ", arr_linear.shape[1], "]")
            print("orientation's shape: [", arr_ori.shape[0], ", ", arr_ori.shape[1], "]")

            if np.array_equal(linear_g, [[]]):
                linear_g = linear
                orientation_g = orientation
            else:
                linear_g = np.vstack((linear_g, linear))
                orientation_g = np.vstack((orientation_g, orientation))

            pdr = pdr.Model(linear_g, orientation_g)
            x, y, steps = pdr.show_trace(frequency=100, walkType='normal', initPosition=(0, 0), offset=0)
            loc = [round(x[steps], 2), round(y[steps], 2)]
            termLoc.append(loc)
            print(loc)
            print("===============")

        response = {'status': 'true', 'TermLoc': loc}

        return response

    def reset(self):
        global x, y, steps, linear_g, orientation_g, termLoc
        termLoc = [[0, 0]]
        x = [0]
        y = [0]
        steps = 0
        linear_g = [[]]
        orientation_g = [[]]


if __name__ == '__main__':
    # 启动HTTP服务，此处需使用电脑打开热点，此时IP就是192.168.137.1，然后使用移动设备连接电脑热点，即可实现通信
    server = HTTPServer(('192.168.137.1', 8001), AndroidHTTPHandler)

    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
