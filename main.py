import torch
from getMap import get_category_map, get_English_tag, get_Chinese_tag
from image_preprocessing import preprocess_image
import socket
from concurrent.futures import ThreadPoolExecutor

# 加载设备与模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("vgg16_caltech256.pth")
model.to(device)
model.eval()

# 得到标签集合
category_map = get_category_map()
English_tag = get_English_tag()
Chinese_tag = get_Chinese_tag()

# socket 服务器配置
HOST = socket.gethostname()  # 获取本地主机名
PORT = 7777  # 端口号
conn_num = 20


# 对于socket连接请求的处理
def handle_client(client_conn, client_addr):
    client_conn.settimeout(10)  # 设置超时时间为 10 秒
    print('接收连接，地址：', client_addr)
    # 接收客户端发送的图片地址
    try:
        data = client_conn.recv(1024)
    except socket.timeout:
        conn.sendall("接收超时")
    print("接收到的网址：")
    print(data.decode())
    # 得到预处理图片,注意末尾有一个换行，需要删掉
    img = preprocess_image(data.decode().strip())
    img = img.to(device)
    # 预测
    with torch.no_grad():
        output = model(img)
    predicted_label = torch.argmax(output).item()
    tag_num = "、" + str(predicted_label) + ";"
    # 拼接结果
    result = "tag:" + Chinese_tag.get(predicted_label) + tag_num
    if category_map.get(predicted_label) is None:
        result = result + "category:未分类"
    else:
        result = result + "category:" + category_map.get(predicted_label)
    print("返回给客户端的消息" + result)
    # 回复客户端消息
    client_conn.sendall(result.encode())
    client_conn.shutdown(socket.SHUT_WR)
    conn.close()


print("服务器开启")
# 使用线程池处理请求
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(conn_num)
    with ThreadPoolExecutor(max_workers=20) as executor:
        while True:
            conn, addr = s.accept()
            executor.submit(handle_client, conn, addr)
