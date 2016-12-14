"""2016003463 남혜민"""
"""TensorFlow를 사용하기 위한 import"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
"""데이터를 자동으로 다운로드하고 설치하는 코드"""
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
"""x : """
x = tf.placeholder(tf.float32, [None, 784])
"""Variable : 가중치와 편향값을 추가적인 입력으로 다루는 것,
    TensorFlow의 상호작용하는 작업 그래프들간에 유지되는 변경 가능한 텐서, 
    계산과정에서 사용되거나 변경 가능
    tf.Variable을 주어 Variable의 초기값을 만듦
    w와 b를 0으로 채워진 텐서들로 초기화
    w: 784차원의 이미지 벡터를 곱하여 10차원 벡터의 증거를 만듦
    b: 출력에 더할 수 있음"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""x와 W를 곱하고 b를 더해 tf.nn.softmax적용"""
"""이때 x*W는 W(x)가 있던 우리 식에서 곱한 결과에서 뒤집혀 있음, 
    x가 여러 입력으로 구성된 2D 텐서일 경우를 다룰 수 있게 하기 위한 것"""
y = tf.nn.softmax(tf.matmul(x, W) + b)
"""y: 우리가 예측한 확률 분포
    y': 실제분포(우리가 입력할 one-hot 벡터)
    교차 엔트로피를 구현하기 위해 정답을 입력하기 위한 새 placeholder를 추가"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""교차 엔트로피 구현
    tf.log: y의 각 원소의 로그 값 계산 
    -> y_의 각 원소들에 해당되는 tf.log(y)를 곱함 
    -> tf.reduce_sum: 텐서의 모든 원소를 더함"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
"""TensorFlow에게 학습도를 0.01로 준 경사 하강법 알고리즘을 이용하여 교차 엔트로피를 최소화하도록 하는 것"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
"""실행 전 변수들을 초기화"""
init = tf.initialize_all_variables()

"""세션에서 모델을 시작하고 변수들을 초기화하는 작업"""
sess = tf.Session()
sess.run(init)

# Learning
"""1000번의 학습을 시킴
    각 반복 단계마다 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리들을 가져옴
    placeholders를 대체하기 위한 일괄 처리 테이터에 train_step 피딩을 실행"""
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
"""맞는 라벨을 예측했는지 확인
    tf.argmax는 특정한 축을 따라 가장 큰 원소의 색인을 알려주는 함수"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
"""얼마나 많은 비율로 맞았는지 확인
    부정소숫점으로 캐스팅한 후 평균값을 구함"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
"""테스트 데이터를 대상으로 정확도를 확인"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))