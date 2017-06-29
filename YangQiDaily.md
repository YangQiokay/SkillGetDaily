注：为个人学习总结需要，如有侵权，请您指出。[持续更新， 如有错误，请您指出]        

> 邮箱：yangqiokay@foxmai.com      微信号：<u>*18810578662*</u>

​    

 ##                      **「礼貌的笑了笑，寸步不让，产出」** 

-----

[TOC]

关于「学术论文／研究项目／工程项目／创业项目」：

### 关于「专业书」：

##### **[《神经网络与深度学习》连载翻译](http://chuansong.me/n/1895437651113)**

- **1.使用神经网络识别手写数字**
  - ​
    - 神经网络（Neural Networks）使用一种不同的思路解决这个问题。它的思想是利用大量的手写数字，亦被称为训练样例
    - 从这些训练样例学习并建立一个系统。换一种说法，神经网络使用这些样例，从中能够自动地学习到识别手写数字的规则。而且，随着训练样例的增加，神经网络可以从中学习到更多信息，从而提高它的准确度。
    - 实现一个神经网络的计算机程序，来学习并识别手写数字。虽然这个程序仅仅只有74行，并且没有使用任何特别的神经网络库，但是它可以在没有任何人工干预的情况下，达到超过百分之96的手写数字识别准确率。
    - **两种重要的人工神经元**（感知机和sigmoid神经元），以及**神经网络的标准学习算法，称为随机梯度下降**（stochastic gradient descent）
  - 感知机
    - 什么是神经网络？在回答这个问题之前，我会先解释一种叫做**感知机（perceptron**）的人工神经元
    - 现如今，我们通常使用其它种类的人工神经元模型——在这本书里，以及在许多关于神经网络的最新工作里，主要使用的是一种叫做**sigmoid神经元（sigmoid neuron）**的神经元模型。
    - 感知机是怎么工作的呢？感知机的输入是几个二进制，x1,x2,x3,…，输出是一位单独的二进制
    - 本例中的感知机有三个输入，x1,x2,x3。通常，它可以有更多或者更少的输入。Rosenblatt提出了一种计算输出的简单的规则。他引入了**权重（weight****）**，w1,w2,…，等实数来表示各个输入对于输出的重要程度。神经元的输出是0还是1，由加权和∑j*wjxj*是否小于或者大于某一个**阈值（threshold value）**。和权重一样，阈值也是一个实数，同时它是神经元的一个参数
    - 这就是感知机的工作方式！
    - 可以这样理解感知机，它是一个通过给evidence赋予不同权重从而作出决策的机器。
    - 显然，感知机不能完全建模人类的决策系统！不过，这个例子阐明的是感知机如何赋予不同evidence权重来达到做出决策的目的。一个由感知机构成的复杂网络能够做出更加精细的决策，这貌似看上去是说得通的：
    - 在这个网络中，第一列感知机——通常称为第一层感知机——通过赋予输入的evidence权重，做出三个非常简单的决策。第二层感知机呢？每一个第二层感知机通过赋予权重给来自第一层感知机的决策结果，来做出决策。通过这种方式，第二层感知机可以比第一层感知机做出更加复杂以及更高层次抽象的决策。第三层感知机能够做出更加复杂的决策。通过这种方式，一个多层网络感知机可以做出更加精细的决策。
    - 我之前定义感知机的时候，我说到感知机只有一个输出。在上面这个网络中，感知机似乎看上去有多个输出。实际上，它们仍然只有一个输出。多个输出箭头仅仅是用来方便表示它的输出被用作其它几个感知机的输入。如果画成一个输出分裂出几个箭头就太难看了。
    - 让我们来简化一下对感知机的描述。∑j*wjxj* > threshold的条件太笨重了，我们可以通过使用两个新记法来简化它。第一是使用点乘来代替∑j*wjxj*。我们有w⋅x≡∑j*wjxj*，其中w和x都是向量，它们的元素分别代表了权重和输入。第二是将阈值移到不等号的另一侧，并使用偏移（bias）来代替阈值threshold， b≡-threshold。于是，感知机规则可以被重写为：
    - 你可以将偏移（bias）理解为感知机为了得到输出为1的容易度的度量。如果从生物的角度来理解，偏移是使神经元被激活的容易度的度量。如果一个感知机的偏移非常大，那么这个感知机的输出很容易为1，相反如果偏移非常小，那么输出1就很困难。
    - 我之前将感知机描述为通过权衡evidence做出决策的一种方法。感知机的另一种用途是计算初等逻辑函数，例如AND、OR和NAND。例如，假如一个感知机有两个输入，每一个权重都是−2，偏移为3
    - 我们可以使用感知机来计算初等逻辑函数。
    - 不要将输入感知机当做感知机，而是理解为一个特殊的单元，它能够输出我们想要的值x1,x2,…
    - 可以设计**学习算法**（learning algorithm）使得能够自动地调整人工神经元网络的权重和偏移。这种调整能够对外部刺激作出响应，而不需要程序员的直接干预。这些学习算法使得我们能够用一种与传统逻辑门从根本上不同的方法使用人工神经元
  - sigmoid神经元
    - 我们该如何为神经网络量身设计一种学习算法呢？现在假设有一个由感知机构成的网络，我们想让这个网络学习如何去解决一些问题。举例来说，对于一个以手写数字的扫描图像的原始像素数据作为输入的网络，我们想要这个网络去学习**权值（weights）**和**偏移（biases）**以便最终正确地分类这些数字。
    - 为了说明学习算法如何才能有效，我们首先假设在网络的一些权值（或偏移）上做一个小的改变
    - 我们期望的结果是，这些在权值上的小改变，将会为网络的输出结果带来相应的改变，且这种改变也必须是轻微的。我们在后面将会看到，满足这样的性质才能使学习变得可能。
    - 如果满足在权值（或偏移）上的小改变只会引起输出上的小幅变化这一性质，那么以此性质为基础，我们就可以改变权值和偏移来使得网络的表现越来越接近我们预期。例如，假设原始的网络会将一张写着「9」的手写数字图片错误分类为「8」。我们可以尝试找到一个正确的轻微改变权值和偏移的方法，来使得我们网络的输出更接近于正确答案——将该图片分类为「9」。重复这个过程，不断地修改权值和偏移并且产生越来越好的结果。这样我们的网络就开始学习起来了。
    - 但问题在于，当我们的网络包含感知机时情况就与上述描述的不同了。事实上，轻微改变网络中任何一个感知机的权值或偏移有时甚至会导致感知机的输出完全翻转——比如说从0变为1。 
    - 这个翻转行为可能以某种非常复杂的方式彻底改变网络中其余部分的行为。所以即使现在「9」被正确分类了，但网络在处理所有其他图片时的行为可能因一些难以控制的方式被彻底改变了。这导致我们逐步改变权值和偏移来使网络行为更加接近预期的学习方法变得很难实施。也许存在一些巧妙的方法来避免这个问题，但对于这种由感知机构成的网络，它的学习算法并不是显而易见的。
    - 由此我们引入一种被称为**S型(sigmoid ，通常我们更习惯使用它的英文称呼，所以本文的其他地方也将使用原始的英文)**神经元的新型人工神经元来解决这个问题。sigmoid神经元与感知机有些相似，但做了一些修改使得我们在轻微改变其权值和偏移时只会引起小幅度的输出变化。这是使由sigmoid神经元构成的网络能够学习的关键因素。
    - 沿用感知机的方式来描述sigmoid神经元。
    - 和感知机一样，sigmoid神经元同样有输入，x1,x2,⋯ ， 但不同的是，这些输入值不是只能取0或者1，而是可以取0到1间的任意浮点值。所以，举例来说，0.638…对于sigmoid神经元就是一个合法输入。同样，sigmoid神经元对每个输入也有相应的权值，w1,w2,…，以及一个整体的偏移，b 。不过sigmoid神经元的输出不再是0或1，而是σ(w⋅x+b) ， 其中的σ被称为******sigmoid函数（sigmoid function）**1
    - 更加准确的定义，sigmoid神经元的输出是关于输入x1,x2,…，权值w1,w2,…，和偏移b的函数
    - 乍一看去，sigmoid神经元与感知机样子很不同。如果你对它不熟悉，sigmoid函数的代数形式看起来会有些晦涩难懂。但事实上，sigmoid神经元与感知机有非常多相似的地方。sigmoid函数的代数式更多地是展现了其技术细节
    - 为了理解sigmoid神经元与感知机模型的相似性，我们假设z≡w⋅x+b是一个很大的正数。这时e−z≈0且σ(z)≈1。即是说，当z=w⋅x+b是一个很大的正数时，sigmoid神经元的输出接近于1，与感知机类似。另一方面，当z=w⋅x+b是一个绝对值很大的负数时，e−z→∞且σ(z)≈0。所以当z=w⋅x+b是一个绝对值很大的负数时，sigmoid神经元的行为与感知机同样很接近。只有当w⋅x+b是一个不太大的数时，其结果与感知机模型有较大的偏差。
    - 的确切形式并不是那么重要——对于我们理解问题，真正重要的是该函数画在坐标轴上的样子。下图表示了它的形状：
    - 个形状可以认为是下图所示**阶梯函数（step function）**的平滑版本：
    - σ函数换成阶梯函数，那么sigmoid神经元就变成了一个感知机，这是因为此时它的输出只随着w⋅x+b的正负不同而仅在1或0这两个离散值上变化2。所以如前面所言，当使用σ函数时我们就得到了一个平滑的感知机。而且，σ函数的平滑属性才是其关键，不用太在意它的具体代数形式。σ函数的平滑属性意味着当我们在权值和偏移上做出值为Δwj，Δb的轻微改变时，神经元的输出也将只是轻微地变化Δoutput
    - Δoutput是关于权值和偏移的改变量Δwj和Δb的**线性函数（linear function）**
    - sigmoid神经元不仅与感知机有很多相似的性质，同时也使描述「输出怎样随权值和偏移的改变而改变」这一问题变得简单
    - σ的形状起作用而其具体代数形式没有什么用的话，为什么公式(3)要把σ表示为这种特定的形式？事实上，在书的后面部分我们也会偶尔提到一些在输出f(w⋅x+b)中使用其它**激活函数（activation function）f(⋅)**的神经元。当我们使用其它不同的激活函数时主要改变的是公式(5)中偏微分的具体值。在我们需要计算这些偏微分值之前，使用σ将会简化代数形式，因为指数函数在求微分时有着良好的性质。不管怎样，σ在神经网络工作中是最常被用到的，也是本书中最频繁的激活函数。
    - 如何解释sigmoid神经元的输出呢？显然，感知机和sigmoid神经元一个巨大的不同在于，sigmoid神经元不仅仅只输出0或者1，而是0到1间任意的实数，比如0.173…，0.689…都是合法的输出。在一些例子，比如当我们想要把神经网络的输出值作为输入图片的像素点的平均灰度值时，这点很有用处。但有时这个性质也很讨厌。比如在我们想要网络输出关于「输入图片是9」与「输入图片不是9」的预测结果时，显然最简单的方式是如感知机那样输出0或者1。不过在实践中我们可以设置一个约定来解决这个问题，比如说，约定任何输出值大于等于0.5的为「输入图片是9」，而其他小于0.5的输出值表示「输入图片不是9」。当以后使用类似上面的一个约定时，我都会明确地说明，所以这并不会引起任何的困惑。
    - σ有时也被称作逻辑斯谛函数（logistic function），对应的这个新型神经元被称为逻辑斯谛神经元（ogistic neurons）
    - 事实上，当w⋅x+b=0时感知机将输出0，但此时阶梯函数输出值为1。
    - **sigmoid神经元模拟感知机（第一部分）**对一个由感知机组成的神经网络，假设将其中所有的权值和偏移都乘上一个正常数，c>0 ， 证明网络的行为并不会发生改变。
  - 神经网络的结构
    - 接下来这部分我将介绍一个能够良好分类手写数字的神经网络。在那之前，解释一些描述网络不同部分的术语是很有必要的。假设我们有如下网络：
    - 网络中最左边的一层被称作**输入层**，其中的神经元被称为**输入神经元（input neurons）**。最右边的一层是**输出层（output layer）**，包含的神经元被称为**输出神经元(output neurons)**。在本例中，输出层只有一个神经元。网络中间的一层被称作**隐层（hidden layer）**，因为它既不是输入层，也不是输出层。「隐」这个字听起来似乎有点神秘的感觉——当我第一次听到这个字时认为其必然包含着深刻的哲学或数学含义——然而除了「既非输入、又非输出」这个含义外，它真的没有别的意思了。
    - 这些多层网络有时又被称为**多层感知机（multilayer perceptron，MLP）**
    - 因为这些网络都是由sigmoid神经元构成的，而非感知机。
    - 网络的输入输出层设计是比较直观的。比如说，假如我们尝试判断一张手写数字图片上面是否写着「9」。很自然地，我们将图片像素的灰度值作为网络的输入。假设图片是64×64的灰度图像，那么我们需要4,096=64×64个输入神经元，每个神经元接受规格化的0到1间的灰度值。输出层只需要包含一个神经元，当输出值小于0.5时说明「输入图片不是9」，否则表明「输入图片是9」。
    - 相对于输入输出层设计的直观，隐层的设计就是一门艺术了。特别的，单纯地把一些简单规则结合到一起来作为隐层的设计是不对的。事实上，神经网络的研究者们已经总结了很多针对隐层的启发式设计规则，这些规则能够用来使网络变得符合预期。举例来说，一些启发式规则可以用来帮助我们在隐层层数和训练网络所需的时间开销这二者间找到平衡
    - 我们讨论的都是**前馈神经网络（feedforward neural networks）**，即把上一层的输出作为下层输入的神经网络。这种网络是不存在环的——信息总是向前传播，从不反向回馈。如果我们要制造一个环，那么我们将会得到一个使σ函数输入依赖于其输出的网络
    - 这很难去理解，所以我们并不允许存在这样的环路。
    - 但是，我们也有一些存在回馈环路可能性的人工神经网络模型。这种模型被称为**递归／循环神经网络（recurrent neural networks）**该模型的关键在于，神经元在变为非激活态之前会在一段有限时间内均保持激活状态
    - 这种激活状态可以激励其他的神经元，被激励的神经元在随后一段有限时间内也会保持激活状态。如此就会导致更多的神经元被激活，一段时间后我们将得到一个级联的神经元激活系统。在这个模型中环路并不会带来问题，因为神经元的输出只会在一段之间之后才影响到它的输入，它并非实时的。
    - 递归神经网络比起前馈神经网络影响力小很多，一部分原因是递归神经网络的学习算法还不够强大，至少目前是如此。不过递归神经网络依然非常有吸引力。从思想上来看它要比前馈神经网络更接近我们大脑的工作方式。而且递归神经网络也可能解决一些重要的、前馈神经网络很难处理的问题。不过为控制篇幅，本书将主要关注更广泛应用的前馈神经网络。
  - 用简单的网络结构解决手写数字识别问题
    - 定义了神经网络之后，让我们回到手写数字识别的问题上来。我们可以把手写数字识别问题拆分为两个子问题
      - 首先，我们要找到一种方法能够把一张包含若干数字的图像分割为若干小图片，其中每个小图像只包含一个数字。举个例子，我们想将下面的图像
      - 当图像被分割之后，接下来的任务就是如何识别每个独立的手写数字。
    - 我们将把精力集中在实现程序去解决第二个问题，即如何正确分类每个单独的手写数字。因为事实证明，只要你解决了数字分类的问题，分割问题相对来说不是那么困难。
    - 分割问题的解决方法有很多。一种方法是尝试不同的分割方式，用数字分类器对每一个切分片段打分。如果数字分类器对每一个片段的置信度都比较高，那么这个分割方式就能得到较高的分数；如果数字分类器在一或多个片段中出现问题，那么这种分割方式就会得到较低的分数。这种方法的思想是，如果分类器有问题，那么很可能是由于图像分割出错导致的。
    - 为了识别数字，我们将会使用一个三层神经网络：
    - 这个网络的输入层是对输入像素编码的神经元。正如我们在下一节将要讨论的，我们的训练数据是一堆28乘28手写数字的位图，因此我们的输入层包含了784=28×28个神经元。
    - 输入的像素点是其灰度值，0.0代表白色，1.0代表黑色，中间值表示不同程度的灰度值。
    - 网络的第二层是隐层。我们为隐层设置了n个神经元，我们会实验n的不同取值。在这个例子中我们只展现了一个规模较小的隐层，它仅包含了n=15个神经元
    - 网络的输出层包含了10个神经元。如果第一个神经元被激活，例如输出≈1，然后我们可以推断出这个网络认为这个数字是0。如果第二层被激活那么我们可以推断出这个网络认为这个数字是1。以此类推。更精确一些的表述是，我们把输出层神经元依次标记为0到9，我们要找到哪一个神经元拥有最高的激活值。如果6号神经元有最高值，那么我们的神经网络预测输入数字是6。对其它的神经元也如此
    - 最终的判断是基于经验主义的：我们可以实验两种不同的网络设计，结果证明对于这个特定的问题而言，10个输出神经元的神经网络比4个的识别效果更好。
    - 们需要从根本原理上理解神经网络究竟在做些什么。首先考虑有10个神经元的情况。我们首先考虑第一个它能那么做是因为可以权衡从隐藏层来的信息。隐藏层的神经元在做什么呢？假设隐藏层的第一个神经元只是用于检测如下的图像是否存在
    - 输出神经元，它告诉我们一个数字是不是0。
    - 为了达到这个目的，它通过对此图像对应部分的像素赋予较大权重，对其它部分赋予较小的权重
    - 如果所有这四个隐藏层的神经元被激活那么我们就可以推断出这个数字是0。当然，这不是我们推断出0的唯一方式——我们能通过很多其他合理的方式得到0 （举个例子来说，通过上述图像的转换，或者稍微变形）。但至少在这个例子中我们可以推断出输入的数字是0 。
    - 假设神经网络以上述方式运行，我们可以给出一个貌似合理的理由去解释为什么用10个输出而不是4个。如果我们有4个输出，那么第一个输出神经元将会尽力去判断数字的最高有效位是什么。把数字的最高有效位和数字的形状联系起来并不是一个简单的问题。很难想象出有什么恰当的历史原因一个数字的形状要素会和一个数字的最高有效位有什么紧密联系。
    - 通过在上述的三层神经网络加一个额外的一层就可以实现按位表示数字。额外的一层把原来的输出层转化为一个二进制表示，如下图所示。为新的输出层寻找一些合适的权重和偏移。假定原先的3层神经网络在第三层得到正确输出（即原来的输出层）的激活值至少是0.99，得到错误的输出的激活值至多是0.01。
  - 通过梯度下降法学习参数
    - 数字实际上就是我们在章节开始所提到的作为挑战去识别的图像。当然，当我们测试我们的神经网络时，我们需要让它去识别不属于训练集中的数据
    - MNIST数据集可以分为两个部分。第一个部分包含了60000个训练图像第二部分是10000个测试图像。同样它们也是28乘28的灰度图像。我们将会用这些测试数据去评估我们的神经网络识别效果如何。
    - 我们将会用x去定义训练输入。将每一个训练输入x视为一个28×28=784-维的向量
    - 量的每一个输入代表一个图像的一个像素点的灰度值。我们定义输出为y=y(x)，每一个y 是一个 10-维的向量
    - 我们需要的是一个算法让我们去找到合适的权重和偏置能让所有的训练输入x都近似于y(x)。
    - 公式里面的w表示所有的权重的集合，b表示所有的偏置，n是训练数据的个数，a代表输出层的向量，这个和是包含了所有的训练输入x。当然输出a 取决于 x, w 和 b，
    - 符号||v||是指向量v的模。我们把C称为二次代价函数；有时我们也称它为均方误差或者MSE
    - 通过代价函数的形式我们可以得知它是非负的，因为加和的每一项都是非负的。另外，代价函数C(w,b)会变小，e.g.C(w,b)≈0，简单来说就是对于所有的训练输入x，我们的输出a将会接近y(x)。因此我们的学习算法能很好的工作如果它能找到合适的权重和偏置能让C(w,b)≈0。
    - 因此我们的训练算法的目标就是通过调整函数的权重和偏执来最小化代价函数C(w,b)
    - 寻找合适的权重和偏置让代价函数尽可能地小。我们将通过**梯度下降法**
    - 为什么要介绍**平方代价**（quadratic cost）呢？毕竟我们最初所感兴趣的内容不是对图像正确地分类么？为什么不增大正确输出的得分，而是去最小化一个像平方代价类似的间接评估呢？这么做是因为在神经网络中，被正确分类的图像的数量所关于权重、偏置的函数并不是一个平滑的函数。大多数情况下，对权重和偏置做出的微小变动并不会影响被正确分类的图像的数量。这会导致我们很难去刻画如何去优化权重和偏置才能得到更好的结果。一个类似平方代价的平滑代价函数能够更好地指导我们如何去改变权重和偏置来达到更好的效果。这就是为何我们集中精力去最小化平方代价，只有通过这种方式我们才能让分类器更精确。
    - 再次回顾一下，我们训练神经网络的目的是寻找合适的权重和偏置来最小化代价函数C(w,b)。
    - 事实证明我们可以忽略大部分问题，把精力集中在最小化方面。
    - 现在我们打算忘掉所有关于代价函数的具体形式，忘掉和神经网络有什么联系。现在想象我们仅仅是要去最小化一个给定的多元函数。我们打算介绍一种可以解决最小化问题的技术-梯度下降法。然后我们回到在神经网络中要最小化的函数上来。
    - 假定我们要最小化某些函数，C(v)。它可能是任意的多元实值函数，v=v1,v2,…。注意我们将会用v去代表w和b以强调它可能是任意的函数-我们不会把问题局限在特定的神经网络上。假定C(v)有两个变量v1 和v2 ： valley 我们想要的就是得到它的全局最小值。
    - 一种解决这个问题的方式就是分析性去计算出它的最小值。我们可以计算出偏导，利用偏导去寻找函数C的极值点。运气好的话我们的函数C只有一个或者少数的几个变量。
    - 在神经网络中我们通常会需要大量的变量-最大的神经网络的代价函数包含了数亿个权重和偏置
    - 为了更精确的描述这个问题，让我们想象一下如果我们在v1方向移动一个很小的量Δv1并在v2方向移动一个很小的量Δv2将会发生什么呢。通过计算可以告诉我们C将会产生如下改变：
    - 我们将要寻找一种方式去选择Δv1和Δv2使得ΔC为负；比如，我们选择它们是为了让小球下落。
    - 为了搞清楚如何选择，有必要定义Δv来代表v变化的向量，
    - ∇C是一个梯度向量
    - 这里的η是个很小的正数（就是我们熟知的学习速率）
    - 那么C将一直会降低，不会增加。（当然，要在（9）式的近似约束下）。这就是我们想要的特性！因此我们把（10）式看做梯度下降算法的“运动定律”。也就是说，我们用（10）式计算Δv，把小球位置v移动
    - 如果我们反复那么做，那么C会一直降低直到我们想要需找的全局最小值。
    - 梯度下降算法工作的方式就是重复计算梯度 ∇C，然后沿着梯度的反方向运动。
    - 为了使我们的梯度下降法能够正确地运行，我们需要选择足够小的学习速率η使得等式（9）能得到很好的近似。如果不那么做，我们将会以ΔC>0结束
    - 这显然不是一个很好的结果。与此同时，我们也不能把η变得过小，因为如果η过小，那么梯度下降算法就会变得异常缓慢。在真正的实现中，η通常是变化的，
    - 已经解释了具有两个变量的函数C的梯度下降法。但事实上当C是其他多元函数时也能很好地运行
    - 我们假设C是一个有m个变量v1,…,vm的多元函数。我们对自变量做如下改变Δv=(Δv1,…,Δvm)T，那么ΔC 将会变为ΔC≈∇C⋅Δv
    - 正如两个变量的例子一样，我们可以选取Δv=−η∇C, 
    - 并且我们也会保证公式[（12）中ΔC是负数
    - 这给了我们一种方式从梯度中去寻找函数的最小值，即使C是任意的多元函数
    - 可以把这个更新规则看做梯度下降算法的定义。这给我们提供了一种方式去通过重复改变v来达到函数的最小值
    - 然而这种规则并不总是有效的-有几件事能导致错误，让我们无法从梯度来求得函数C的全局最小值
    - 从实践方面来看，梯度下降法通常效果非常好，在神经网络中这是一种非常有效的方式去求代价函数的最小值，进而促进网络自身的学习
    - 有一种观念认为梯度下降法是求最小值的最优策略。我们假设努力去改变Δv来让C尽可能地减小，减小量为ΔC≈∇C⋅Δv。我们首先限制步长为固定值，即||Δv||=ϵ ，ϵ>0。当步长固定时，我们要找到使得C减小最大的下降方向。可以证明，使得∇C⋅Δv取得最小值的Δv为Δv=−η∇C，这里η=ϵ/||∇C||是由步长限制||Δv||=ϵ所决定的。因此，梯度下降法可以被视为一种通过在C下降最快的方向上做微小变化来使得C立即下降的方法。
    - 有种叫做随机梯度下降(SGD) 的算法能够用来加速学习过程
    - 想法就是通过随机选取小量输入样本来计算∇Cx，进而可以计算∇C。采取少量样本的平均值可以快速地得到梯度∇C，这会加速梯度下降过程，进而加速学习过程。
    - 更准确地说，SGD是随机地选取小量的m个训练数据。我们将选取的这些训练数据标号X1,X2,…,Xm，并把它们称为一个mini-batch。我们选取的m要足够大才能保证∇CXj的平均值才能接近所有的平均值∇Cx
    - SGD就是随机地选取大小为一个mini-batch的训练数据，然后去训练这些数据，
    - 关于调节代价函数和mini-batch去更新权重和偏置有很多不同的约定。
    - 使用小规模的mini-batch计算梯度，比使用整个训练集计算梯度容易得多，
    - 我们只关心在某个方向上移动可以减少C，这意味着我们没必要准确去计算梯度的精确值
    - 梯度下降一个比较极端的版本就是让mini-batch的大小变为1。
    - 当我们选取另一个训练数据时也做同样的处理。其他的训练数据也做相同的处理。这种处理方式就是**online learning**
    - 神经网络在一个时刻只学习一个训练数据
    - 在神经网络中，代价函数C是一个关于所有权重和偏置的多元函数，因此在某种意义上来说，就是在一个高维空间定义了一个平面。
  - 实现我们的网络来分类数字

    - 写一个学习怎么样识别手写数字的程序，使用随机梯度下降法和MNIST训练数据。我们需要做的第一件事情是获取MNIST数据
    - 我们将测试集保持原样，但是将60,000个图像的MNIST训练集分成两个部分：一部分50,000个图像，我们将用来训练我们的神经网络，和一个单独的10,000个图像的**验证集（validation set）**
    - 验证数据，但是在本书的后面我们将会发现它对于解决如何去设置神经网络中的**超参数（hyper-parameter）**——例如学习率等不是被我们的学习算法直接选择的参数——是很有用的。
    - 尽管验证数据集不是原始MNIST规范的一部分，然而许多人使用以这种方式使用MNIST，并且在神经网络中使用验证数据是很常见的。当我从现在起提到「MNIST训练数据」，我指的不是原始的60,000图像数据集，而是我们的50,000图像数据集
    - 这段代码中，列表sizes包含各层的神经元的数量
    - 因此举个例子，如果我们想创建一个在第一层有2个神经元，第二层有3个神经元，最后层有1个神经元的network对象，我们应这样写代码：   net = Network([2, 3, 1])
    - Network对象的偏差和权重都是被随机初始化的，使用Numpy的np.random.randn函数来生成均值为0，标准差为1的高斯分布
    - 随机初始化给了我们的随机梯度下降算法一个起点。在后面的章节中我们将会发现更好的初始化权重和偏差的方法，但是现在将采用随机初始化。
    - 注意Network初始化代码假设第一层神经元是一个输入层，并对这些神经元不设置任何偏差
    - 同样注意，偏差和权重以列表存储在Numpy矩阵中。因此例如**net.weights[1]**是一个存储着连接第二层和第三层神经元权重的Numpy矩阵。（不是第一层和第二层，因为Python列中的索引从0开始。）因此net.weights[1]相当冗长， 让我们就这样表示矩阵w。矩阵中的wjk是连接第二层的kth神经元和第三层的jth神经元的权重。这种j和k索引的顺序可能看着奇怪，当然交换j和k索引会更有意义？使用这种顺序的很大的优势是它意味着第三层神经元的激活向量是
    - a是第二层神经元的激活向量。为了得到a′，我们用权重矩阵w乘以a，加上偏差向量b，我们然后对向量wa+b中的每个元素应用函数σ。**（这被叫做函数σ的向量化（vectorizing）。）**很容易验证等式（22）给出了跟我们之前的计算一个sigmoid神经元的输出的等式
    - 注意，当输入z是一个向量或者Numpy数组时，Numpy自动的应用元素级的sigmoid函数，也就是向量化。
    - training*data是一个代表着训练输入和对应的期望输出的元组（x,y）的列表。变量epochs和mini*batch*size是你期望的训练的迭代次数和取样时所用的mini-batch块的大小。eta是学习率η。*
    - 这行调用了一个叫做叫**反向传播（backpropagation）**的算法，这是一种快速计算代价函数的梯度的方法。
    - 我不得不对于训练的迭代次数，mini-batch大小和学习率η做了特殊的选择。正如我上面所提到的，这些在我们的神经网络中被称为超参数，以区别于通过我们的学习算法所学到的参数（权重和偏置）
  - 迈向深度学习
    - 虽然我们的神经网络给出了令人印象深刻的表现，但这个过程有一些神秘。网络中的权重和偏置是自动被发现的。这意味着我们不能对神经网络的运作过程有一个直观的解释。
    - 设计出了一个网络，它将一个非常复杂的问题——这张图像是否有一张人脸——分解成在单像素层面上就可回答的非常简单的问题。在前面的网络层，它回答关于输入图像非常简单明确的问题，在后面的网络层，它建立了一个更加复杂和抽象的层级结构。包含这种多层结构（两层或更多隐含层）的网络叫做**深度神经网络（deep neural networks）**
    - 在网络中通过人工来设置权重和偏置是不切实际的。取而代之的是，我们使用学习算法来让网络能够自动的从训练数据中学习权重和偏置
    - 些深度学习技术基于随机梯度下降和反向传播，并引进了新的想法。这些技术能够训练更深（更大）的网络——现在训练一个有5到10层隐层的网络都是很常见的。而且事实证明，在许多问题上，它们比那些仅有单个隐层网络的浅层神经网络表现的更加出色。

- 2. 反向传播是如何工作的

  - ​
    - 学习了神经网络是如何利用梯度下降算法来学习权重（weights）和偏置（biases）的。然而，在我们的解释中跳过了一个细节：**我们没有讨论如何计算损失函数的梯度**。这真是一个巨大的跳跃！在本章中我会介绍一个快速计算梯度的算法，就是广为人知的**\*反向传播算法***（backpropagation algorithm）
    - 论文介绍了几种神经网络，在这些网络的学习中，反向传播算法比之前提出的方法都要快。这使得以前用神经网络不可解的一些问题，现在可以通过神经网络来解决。今天，反向传播算法是神经网络学习过程中的关键（workhorse）所在
    - 想要理解神经网络，你必须了解其中的细节。反向传播算法的核心是一个偏微分表达式：
    - 表示损失函数C对网络中的权重w（或者偏置b）求偏导。这个式子告诉我们，当我们改变权重和偏置的时候，损失函数的值变化地有多快
    - 反向传播不仅仅是一种快速的学习算法，它能够让我们详细深入地了解改变权重和偏置的值是如何改变整个网络的行为的
  - 热身：一个基于矩阵的快速计算神经网络输出的方法
    - 在讨论反向传播算法之前，我们先介绍一个基于矩阵的快速计算神经网络输出的方法来热热身
    - 先介绍一种符号来表示网络中的权重参数，这种表示法不会引发歧义。我们用w*l**jk*来表示从第*l*−1层的第*k*个神经元到第*l*层的第*j*个神经元的连接的权重。例如，下图展示了从第二层的第四个神经元到第三层的第二个神经元的连接的权重
    - 一个奇怪之处就是*j*和*k*下标的顺序。你可能会认为用*j*表示输入神经元，用*k*表示输出神经元更加合理。然而并不是这样的。
    - 我们用一种相似的记法来表示网络的偏置和激活值。确切地，我们用b*l**j*表示第*l*层第*j*个神经元的偏置，用a*l**j*表示第*l*层的第*j*个神经元的激活值
    - 利用这些记法，第*l*层第*j*个神经元的激活值a*l**j*通过下面的式子与第*l*−1层神经元的激活值联系起来
    - 这里的求和针对的是第*l*−1层的所有神经元*k*。为了将这个式子写成矩阵形式，我们为每一层*l*定义一个权重矩阵w*l*。矩阵w*l*的每一项就是连接到第*l*层神经元的权重，这就是说，w*l*中第*j*行第*k*列的元素就是w*l**jk*
    - 最后一个我们需要改写成矩阵形式的想法就是把σ这样的函数向量化。我们在上一章中简单提到了向量化，这里面的思想就是我们想把一个函数比如σ，应用到一个向量v中的每一项。
  - 关于损失函数的两个假设
    - 反向传播算法的目标是计算代价函数C对神经网络中出现的所有权重w和偏置b的偏导数∂C/∂w和∂C/∂b。为了使反向传播工作，我们需要对代价函数的结构做两个主要假设。
    - 反向传播实际上是对单个训练数据计算偏导数∂Cx/∂w和∂Cx/∂b。然后通过对所有训练样本求平均值获得∂C/∂w和∂C/∂b。事实上，有了这个假设，我们可以认为训练样本x是固定的，然后把代价Cx去掉下标表示为C。
    - 这是一个关于输出激活值的函数。显然，该代价函数也依赖于期望的输出y，所以你可能疑惑为什么我们不把代价视为关于y的函数。记住，输入的训练样本x是固定的，因此期望的输出y也是固定的。
  - Hadamard积
    - 反向传播算法是以常见线性代数操作为基础——诸如向量加法，向量与矩阵乘法等运算。但其中一个操作相对不是那么常用。
    - 具体来讲，假设s和t是两个有相同维数的向量。那么我们用s⊙t来表示两个向量的**对应元素(elementwise)**相乘
    - 这种对应元素相乘有时被称为**Hadamard积（Hadamard product）**或**Schur积(Schur product)**。我们将称它为Hadamard积
    - 优秀的矩阵库通常会提供Hadamard积的快速实现，这在实现反向传播时将会有用
  - 反向传播背后的四个基本等式
    - 反向传播(backpropagation)能够帮助解释网络的权重和偏置的改变是如何改变代价函数的。
    - 归根结底，它的意思是指计算偏导数∂C/∂wljk和∂C/∂blj。但是为了计算这些偏导数，我们首先介绍一个中间量，δlj，我们管它叫做第l层的第j个神经元的错误量(error)。
    - 反向传播会提供给我们一个用于计算错误量的流程，能够把δlj和∂C/∂wljk,∂C/∂blj关联起来。
    - 为了理解错误量是如何定义的，想象一下在我们的神经网络中有一个恶魔
    - 个恶魔位于第l层的第j个神经元。当神经元的输入进入时，这个恶魔扰乱神经元的操作。它给神经元的加权输入添加了一点改变Δzlj，这就导致了神经元的输出变成了σ(zlj+Δzlj)，而不是之前的σ(zlj)。这个改变在后续的网络层中传播，最终使全部代价改变了∂C/∂zlj*Δzlj。
    - 这个恶魔变成了一个善良的恶魔，它试图帮助你改善代价，比如，它试图找到一个Δzlj能够让代价变小。假设∂C/∂zlj是一个很大的值（或者为正或者为负）。然后这个善良的恶魔可以通过选择一个和∂C/∂zlj符号相反的Δzlj使得代价降低。相比之下，如果∂C/∂zlj接近于0，那么这个恶魔几乎不能通过扰乱加权输入zlj改善多少代价。
    - 这只是针对很小的Δzlj来说的。我们会做出假设来限制恶魔只能做出很小的变化。
    - 受到这个故事的促动，我们定义l层第j个神经元的错误量δlj为
    - 按照通常的习惯，我们使用δl来表示与l层相关联的错误量的向量。反向传播将会带给我们一个计算每一层δl的方法，然后把这些错误量联系到我们真正感兴趣的量：∂C/∂wljk和∂C/∂blj。
    - 你可能想知道为什么恶魔一直在更改加权输入zlj。的确，更加自然的想法是，这个恶魔更改的是输出激活量alj，然后我们就可以使用∂C/∂alj来衡量错误。事实上如果你这么做，就会得到和之后的讨论十分相似的结果，但结果会使反向传播的代数形式变得复杂。所以我们会继续使用δlj=∂C/∂zlj用于衡量错误
    - 术语”错误量（error）”通常用于表示分类的错误率（failure rate）。例如，如果神经网络的数字分类准确率是96.0%，那么错误率就是4.0%。
    - 反向传播基于四个基本等式。这些等式带给我们一个计算错误量δl以及代价函数的梯度的方法
    - 下面我会说明这四个等式。尽管如此，我在这里要告诫各位一下：不要期望立刻吸收理解这四个等式。你的这种期望会带失望的
    - 事实上，反向传播的这几个等式内容很多，理解它们需要一定的时间和耐心，随着你会逐渐深入的探索才会真正理解。
    - 现在先预览一下这些方法：我将给出这些等式的简短证明，帮助解释为什么它们是正确的；我们将要以伪代码的算法形式重新阐述一下这些等式，然后再看一下这些伪代码是如何通过Python代码实现的；在本章的最后一部分，我们将会开发一个直观的图片用来解释反向传播的这些等式是什么意思、一个人如何从零开始发现这些等式。在这些过程中，我们会重复提到四个基本等式，这样你也会加深对这些等式的理解，它们会变得舒服，甚至有可能变得漂亮而且自然。
    - **输出层中关于错误量δL的等式**
    - 这是一种非常自然的表达。右侧的第一项∂C/∂aLj，就是用于测量第j个输出激活代价改变有多快的函数。举个例子，如果C并不太依赖于某个特别的输出神经元j，那么δLj就会很小，这是我们所期望的。右侧的第二项σ′(zLj)，用于测量zLj处的激活函数σ改变有多快。
    - 等式(BP1)是δL的分量形式。它是一个完美的表达式，但并不是我们想要的基于矩阵的形式，那种矩阵形式可以很好的用于反向传播。然而，我们可以很容易把等式重写成基于矩阵的形式，就像：
    - ∇aC是一个向量，它是由∂C/∂aLj组成的。你可以把∇aC看做现对于输出激活的C的改变速率。很容易看出来等式(BP1)和(BP1a)是等价的
    - **依据下一层错误量δl+1获取错误量δl的等式**
    - 当我们使用转置权值矩阵(wl+1)T的时候，我们可以凭借直觉认为将错误反向（backward）移动穿过网络，带给我们某种测量第l层输出的错误量方法。然后我们使用Hadamard乘积⊙σ′(zl)。这就是将错误量反向移动穿过l层的激活函数，产生了l层的加权输入的错误量δl
    - 如果输入神经元是低激活量的，或者输出神经元已经饱和（高激活量或低激活量），那么权重就会学习得缓慢。
  - 4个基本方程的证明
    - 现在我们将证明四个基本方程（BP1）-（BP4）。这四个方程都是多元积分链式法则的的结果。如果你对链式法则
    - 我们从方程（BP1）开始讲解，该方程给出了输出错误量的表达式δL
    - 链式法则, 我们可以把上述偏导数重新表达成带有输出激活的偏导数形式
    - 这里的求和是对于输出层的所有神经元k而言的。当然，当k=j时，输出激活 aLK第kth个神经元只依赖于对第jth个神经元的输入权重zLj。并且当k≠j时，∂aLk/∂zLj项就没有了。因此我们可以简化之前的方程为
    - 再证 (BP2)，它给出了根据下一层的错误量δl+1计算δl的等式。为证明该等式，我们先依据δkl+1=∂C/∂zkl+1重新表达下等式
    - 最后一行，我们互换了下表达式右侧的两项，并取代了 δkl+1的定义。为了评估最后一行的第一项，注意
    - 反向传播的四个基本方程的证明
    - 证明看似复杂，但其实就是仔细应用链式法则得到的结果。简洁点说,我们可以把反向传播看作系统地应用多元变量微积分的链式法则计算代价函数梯度的方法。 这就是所有的反向传播内容，剩下的就是一些细节问题了
  - 反向传播
    - 反向传播等式为我们提供了一个计算代价函数梯度的方法。下面让我们明确地写出该算法：
    - **输入 ![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2pPTwXOH933a6icvpxFg6npeRK9LibgAzzbo2f4LXv5jt1HzQribqbPxlibA/0?wx_fmt=png):**计算输入层相应的激活函数值![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2p1bSianfu1LNqJAgI9xAEaibac7zPupxiatK09y8HPykxoJDyD474M5k2A/0?wx_fmt=png)。
    - **正向传播**
    - **输出误差**
    - **将误差反向传播**
    - **输出**
    - 算法就能看出它为什么叫**反向**传播算法
    - 反向计算错误向量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2plxLibQ5Y8nhYopyvnjacWs7f0VH1ibqnblaDxhfQeLyVhlCOsZEc3bqA/0?wx_fmt=png)。在神经网络中反向计算误差可能看起来比较奇怪。但如果回忆反向传播的证明过程，会发现反向传播的过程起因于代价函数是关于神经网络输出值的函数。为了了解代价函数是如何随着前面的权重和偏移改变的，我们必须不断重复应用链式法则，通过反向的计算得到有用的表达式。
    - 假设我们修改了正向传播网络中的一个神经元，使得该神经元的输出为
    - 在随机梯度下降算法中，我们需要计算一批训练样本的梯度。给定一小批(mini-batch)![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2plyDXiakANADYWQ13W5M1DX5qibykmO4xFwlqkTU4IaLAdRY7u4ykSteA/0?wx_fmt=png)个训练样本
  - 反向传播算法代码
    - 可以理解上一章中用来实现反向传播算法的代码了。
    - **在一个批次（mini-batch）上应用完全基于矩阵的反向传播方法**
    - 在我们的随机梯度下降算法的实现中，我们需要依次遍历一个批次（mini-batch）中的训练样例。我们也可以修改反向传播算法，使得它可以同时为一个批次中的所有训练样例计算梯度。我们在输入时传入一个矩阵![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2pCQFxtzHZJmThicRsqpV7dpMbrE3ibqFUll4OMdBiavsdISgoCKbaufKKw/0?wx_fmt=png)（而不是一个向量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjrGttTtrib652ycGF2vVgw2pPTwXOH933a6icvpxFg6npeRK9LibgAzzbo2f4LXv5jt1HzQribqbPxlibA/0?wx_fmt=png)），这个矩阵的列代表了这一个批次中的向量
    - 明确地写出这种反向传播方法，并修改`network.py`，令其使用这种完全基于矩阵的方法进行计算。
    - 在实践中，所有正规的反向传播算法库都使用了这种完全基于矩阵的方法或其变种
  - 为什么反向传播算法很高效
    - 第一个想到使用梯度下降方法来进行训练的人！但是要实现这个想法，你需要一种计算代价函数梯度的方式
    - 链式法则来计算梯度。但是琢磨了一会之后发现，代数计算看起来非常复杂，你也因此有些失落。所以你尝试着寻找另一种方法。你决定把代价单独当做权重的函数
    - 我们可以利用相同的思想来对偏置求偏导
    - 设想在我们的神经网络中有一百万个权重，对于每一个不同的权重![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjooEia8r1bdwicA62RZ2uddknCeagnMMYl4CiclpjjXwbwo6LQVicPROueGCKzAEpibibwmIvXC3nrW1GXg/0?wx_fmt=png)，为了计算![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjooEia8r1bdwicA62RZ2uddknibxwVGUsicTxOdgbI8pdOX32J0HibJAsaW9LEOwhDPbdVpEubUxZicWtAg/0?wx_fmt=png)，我们需要计算![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjooEia8r1bdwicA62RZ2uddknNxjVvXHnYOfvx5j152diayNwC2qn3HzzAAg0rODiaQemcDr84SNZFw6g/0?wx_fmt=png)。这意味着为了计算梯度，我们需要计算一百万次代价函数，进而对于每一个训练样例，都需要在神经网络中前向传播一百万次
    - 反向传播的优点在于它仅利用一次前向传播就可以同时计算出所有的偏导![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjooEia8r1bdwicA62RZ2uddknibxwVGUsicTxOdgbI8pdOX32J0HibJAsaW9LEOwhDPbdVpEubUxZicWtAg/0?wx_fmt=png)，随后也仅需要一次反向传播
    - 大致来说，反向传播算法所需要的总计算量与两次前向传播的计算量基本相等（这应当是合理的，但若要下定论的话则需要更加细致的分析。合理的原因在于前向传播时主要的计算量在于权重矩阵的乘法计算，而反向传播时主要的计算量在于权重矩阵转置的乘法
    - 极大地拓展了神经网络能够适用的范围，也导致了神经网络被大量的应用。当然了，反向传播算法也不是万能的。在80年代后期，人们终于触及到了性能瓶颈，在利用反向传播算法来训练深度神经网络（即具有很多隐含层的网络）时尤为明显
  - 反向传播：整体描述
    - 反向传播涉及了两个谜题。第一个谜题是，**这个算法究竟在做什么？**我们之前的描述是将错误量从输出层反向传播。但是，我们是否能够更加深入，对这些矩阵、向量的乘法背后作出更加符合直觉的解释
    - 第二个谜题是，**人们一开始是如何发现反向传播算法的？**按照算法流程一步步走下来，或者证明算法的正确性，这是一回事
    - 但这并不代表你能够理解问题的本质从而能够从头发现这个算法。是否有一条合理的思维路线使你能够发现反向传播算法？在本节中，我会对这两个谜题作出解释。
    - 为了更好地构建的反向传播算法在做什么的直觉，让我们假设我们对网络中的某个权重![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0J3uyFCg03Pib6yam1WjNEhuZP1IHsnna6aMJiccw050T02k2japrj7yw/0?wx_fmt=png)做出了一个小的改变量
    - 改变量会导致与其相关的神经元的输出激活值的改变
    - 推，会引起下一层的**所有**激活值的改变
    - 些改变会继续引起再下一层的改变、下下层…依次类推，直到最后一层，然后引起代价函数的改变：
    - 代价函数的改变量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY01iaZGIkeG9UD9T16u0pGSGja6BzGicey7NvYibFtHlCedMAh9wAY1vDfw/0?wx_fmt=png)与最初权重的改变量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0uq7lUTYUfsflHvh30xPfvBqY6Fs8OjicUnuiaAVkb6OwflheqzddK10w/0?wx_fmt=png)是有关
    - 可能方法是，计算![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0J3uyFCg03Pib6yam1WjNEhuZP1IHsnna6aMJiccw050T02k2japrj7yw/0?wx_fmt=png)上的一个小改变量经过正向传播，对![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0BgZjmicSZCQ4ntuPlunEkVkFEVEGWahfLxk6eBiahW493rYdu6gEOMdw/0?wx_fmt=png)引起了多大的改变量。如果我们能够通过小心翼翼的计算做到这点，
    - 激活值改变量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0XsWNfIghUreuDwUvZeSsSequHKu91jGmK92kYFyrCfibnOk0riafwicXQ/0?wx_fmt=png)会引起下一层（第![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0jNQqntJgTicwKrLavFxH0K152N9GzDvYCHhBIiaGLHyo765Or8aJjzbw/0?wx_fmt=png)层）的所有激活值都改变。我们先关注其中的一个结点
    - 当然，改变量![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0IE9ribZxqc7lxjBGVh0thjEEXeL7nbDRu7JzzOn8wx0icP5FORYqgUwQ/0?wx_fmt=png)会继续造成下一层的激活值的改变。实际上，我们可以想象一条从![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0J3uyFCg03Pib6yam1WjNEhuZP1IHsnna6aMJiccw050T02k2japrj7yw/0?wx_fmt=png)到![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0BgZjmicSZCQ4ntuPlunEkVkFEVEGWahfLxk6eBiahW493rYdu6gEOMdw/0?wx_fmt=png)的路径，其中每一个结点的激活值的改变都会引起下一层的激活值的改变，最终引起输出层的代价的改变。
    - 这就计算出了在神经网络的这条路径上，最初的改变量引起了![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0BgZjmicSZCQ4ntuPlunEkVkFEVEGWahfLxk6eBiahW493rYdu6gEOMdw/0?wx_fmt=png)多大的改变。
    - 我们现在只考虑了其中的一条路径。为了计算![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0BgZjmicSZCQ4ntuPlunEkVkFEVEGWahfLxk6eBiahW493rYdu6gEOMdw/0?wx_fmt=png)最终总共的改变量，很显然我们应该对所有可能的路径对其带来的改变量进行求和：
    - 不过，它在直觉上很容易理解。我们计算出了![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjoCvuPCxmKStpodBKf5NkY0BgZjmicSZCQ4ntuPlunEkVkFEVEGWahfLxk6eBiahW493rYdu6gEOMdw/0?wx_fmt=png)相对于网络中的一个权重的变化速率
    - 这个等式告诉我们的是，每一条连接两个神经元的边都可以对应一个变化速率，这个变化速率的大小是后一个神经元对前一个神经元的偏导。连接第一个权重和第一个神经元的边对应的变化速
    - 从起始权重到最终代价上的所有可能的路径的变化速率的总和。
    - 目前为止我所阐述的是一种启发式的观点，当你困惑于神经网络中的权重时，可以通过这种观点来思考。
    - 你可以将反向传播算法看作是一种对所有路径上的所有变化率进行求和的方法。
    - 或者用另外一种方式来说，反向传播算法是一种很聪明的方法

- 3.改进神经网络的学习方法

  - 改进神经网络的学习方法
    - 高尔夫运动员刚开始接触高尔夫时，他们通常会花费大量的时间来练习基本的挥杆。只有不断矫正自己的挥杆方式，才能学好其它的技能：切球，打左曲球，右曲球。同理，到目前为止我们把精力都放在了理解反向传播算法（backpropagation algorithm）上。这是我们的最基本的“挥杆方式”，它是大多数神经网络的基础。在本章中我将介绍一套技术能够用于改进朴素的反向传播算法，进而能改进神经网络的学习方式。
    - 选取更好的代价函数，就是被称为**交叉熵代价函数（the cross-entropy cost function）**；四种**正则化**方法（L1和L2正则、dropout、训练数据的扩展），这能让我们的网络适应更广泛的数据；一种更好的初始化权重（weight）的方法；一系列更具启发式的方法用来帮助我们选择网络的超参数（hyper-parameters）。我也会简单介绍一下其它的技术。这些讨论的内容之间是独立的，因此你可以跳跃着阅读。我们也会用代码实现这些技术去改进我们第一章的手写数字识别的结果。
  - 交叉熵损失函数
    - 我当时非常尴尬。尽管犯错时很不愉快，但是我们能够从明显的错误中学到东西
    - 你能猜到在我下次弹奏的时候会把这个八度弹对。相反，如果错误很不明显的话，我们的学习速度将会很慢。
    - 我们希望神经网络能够快速地从错误中学习。这种想法现实么？为了回答这个问题，让我们看一个小例子。这个例子包含仅有一个输入的单个神经元
    - 输出![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo1y1KFJGgRqLZrPQWiccbRtrjsA5mYQXFBUubkibShnyvMHSvdeIeZhicg/0?wx_fmt=png)。当然，这是一个非常容易的任务，我们可以不利用任何学习算法，通过手算就能找到合适的权重（weight）和偏移（bias）
    - 尽管如此，事实证明利用梯度下降法（gradient descent）能够帮助我们去学习权重和偏移。那么我们就来看一下这个神经元是如何学习的。
    - 我将为权重选定初始值![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoMqsZkZU9DpQPWgnT0iasicsXibX2WJA2zzxGUV4bVRY5sEbjAqoRFiaMmQ/0?wx_fmt=png)，为偏移选定初始值![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo5LlHqmibF1w9ibO8H6LJbbS9DJ2SjdByqB0omqMIa2WribDOZr2lwuRYw/0?wx_fmt=png)。这是学习算法开始时一般的初始选择，我没有用到什么特殊的方式来选取这些初始值。神经元的第一次输出为![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoShezEI8uSynIzUSUBibJgoR9HRaGiaEFicIYzJtARMicX4DhM7Yz3ZArJQ/0?wx_fmt=png)，在到达我们的期望值![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo1y1KFJGgRqLZrPQWiccbRtrjsA5mYQXFBUubkibShnyvMHSvdeIeZhicg/0?wx_fmt=png)之前，神经元还需要很多轮学习迭代
    - 神经元能够迅速地学习权重和偏移来降低代价函数，并最后给出大概![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPom2KYiaKxBoYQuGATGYLiahlPgf5ZRPMeubfzYDcWiblibicE8vhopbHZ7FQ/0?wx_fmt=png)左右的输出。
    - 虽然这不是我们期待的输出，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo63NpFbNY3gBic6tplhrNQbRmKX3yiahibibpouKaVMQahx30UeRVodhGtg/0?wx_fmt=png)，但这个结果已经足够好了。假设我们把权重和偏移的初始值都选为
    - 在这种情况下，初始的输出是![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoNiblWfFTv5icUYmNLrgD52Bbk4X2UVjwyCoxCGC0p3BUgt8QB02yOAmA/0?wx_fmt=png)，这是相当糟糕的结果。让我们看一下在这个例子中神经元是如何学
    - 我们常常能够在错误很大的情况下能学习地更快。但是正如刚才所见，我们的人工神经元在错误很大的情况下学习遇到了很多问题。另外，事实证明这种行为不仅在这个简单的例子中出现，它也会在很多其他的神经网络结构中出现。为什么学习变慢了呢？我们能找到一种方法来避免这种情况么？
    - 我们来考虑一下神经元的学习方式：通过计算代价函数的偏导![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoqicIMucSLKMAWFlw0wjPWUmOeJNXcg7ruwxAYO1E0QrwZEw6CMzVJjw/0?wx_fmt=png)和![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo8Po07aQatYfx8wbBdZtLm3icmaBeopC3vpvq8KAibCOh24dvWZCQlic5g/0?wx_fmt=png)来改变权重和偏移
    - 那么我们说「学习速度很慢」其实上是在说偏导很小。那么问题就转换为理解为何偏导很小
    - 下面我们用权重和偏移来重写这个式子
    - 这种情况导致的速度下降不仅仅适应我们的示例神经元网络，它还适用于很多其他通用的神经元网络
    - 我们可以用不同的代价函数比如交叉熵（cross-entropy）代价函数来替代二次代价函数
    - 为了理解交叉熵，我们暂时先不用管这个示例神经元模型。我们假设要训练一个拥有多个输入变量的神经元：输入![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoy19REvN7iaNEhwaBK8BR5FWdDicibibCPF5DBib8bY9RQpHDDOm4QkZXk4g/0?wx_fmt=png)，权重![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoUpiamcQqHE4FHV7Db57icAiaXcsondyvXudOejrGZGiaBrfyWZq1Nic8hMg/0?wx_fmt=png)，偏移![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPor3uuvCFYy5KBB5pfI4woibCNb6IPSKQHa4NibQicQcRSX0LweaIcNRu2g/0?wx_fmt=png):
    - 这里![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoJCJ4HGvDNcGI1B70CiaZUU2CSehZVjvn4FBEZibfCd7UZgpG8ETptusQ/0?wx_fmt=png)是训练数据的个数，这个加和覆盖了所有的训练输入![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPokk0JIFO8ogl7ianAnK8NtStP5icZ47yrcp4uKicBh0Magex2UNtqSEibkQ/0?wx_fmt=png)，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo21q1ZibJXb8E4ibAsUTugUCrvWbgnibVibrTAIuhT2kAuwOmQlLWdVNqzg/0?wx_fmt=png)是期望输出
    - 仅从等式（57）我们看不出为何能解决速度下降的问题。
    - 交叉熵有两个特性能够合理地解释为何它能作为代价函数。首先，它是非负的，也就是说，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoN5xW20qZ8sV0KScUNmq0ZP4LjHeia5EiaFiaLn8dcJiajzwRcBrJHUwcNA/0?wx_fmt=png)。为了说明这个，我们需要注意到：(a)等式（57）加和里的每一项都是负的，因为这些数是![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo1y1KFJGgRqLZrPQWiccbRtrjsA5mYQXFBUubkibShnyvMHSvdeIeZhicg/0?wx_fmt=png)到![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoPHqS23iaepueGe2ttJUPNXr4UKlqmdRGMGyMRoCdK7BpGUO2kjW7OYQ/0?wx_fmt=png)之间的，它们的对数是负的；(b)整个式子的前面有一个负号。
    - 其次，如果对于所有的训练输入![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPokk0JIFO8ogl7ianAnK8NtStP5icZ47yrcp4uKicBh0Magex2UNtqSEibkQ/0?wx_fmt=png)，这个神经元的实际输出值都能很接近我们期待的输出的话，那么交叉熵将会非常接近0。为了说明这个，假设有一些输入样例![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPokk0JIFO8ogl7ianAnK8NtStP5icZ47yrcp4uKicBh0Magex2UNtqSEibkQ/0?wx_fmt=png)得到的输出是![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoWbaoWGxg7TqmcR9wmzkfziaGcPSPbsgBWciaSwVPM2o77fwoyeZEedmA/0?wx_fmt=png)，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo5aW58yArMDXWaeyDXZvrPhibgLkkmKiass5wKR6IiaoEibicMj39MxSibXlA/0?wx_fmt=png)。这些都是一些比较好的输出。我们会发现等式（57）的第一项将会消掉，因为![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPolNx4ZZBT1icGS8ssbRHMILKGBz856MPFUPw8KRj5cYibw52VrTqvPH2A/0?wx_fmt=png)，与此同时，第二项![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPohOKbsAMDTpaewNyrhzZJk7X9TAzIOiaGcJ7rXSE9Y1UXgnXzLuBeibGw/0?wx_fmt=png)。同理，当![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoQPHic3ibMI6QvstCTGwa0okdwwicw5jaKNibibSAMB6kBdXLic3NL8eUJHjw/0?wx_fmt=png)或![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPojscBoR0NLyf3Ck7kfib5ev5fHgtF6OObSicQGKicPMichg31fTUxNCVLiaw/0?wx_fmt=png)时也如此分析。那么如果我们的实际输出接近期望输出的话代价函数的分布就会很低
    - 交叉熵是正的，并且当所有输入![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPokk0JIFO8ogl7ianAnK8NtStP5icZ47yrcp4uKicBh0Magex2UNtqSEibkQ/0?wx_fmt=png)的输出都能接近期望输出![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo21q1ZibJXb8E4ibAsUTugUCrvWbgnibVibrTAIuhT2kAuwOmQlLWdVNqzg/0?wx_fmt=png)的话，交叉熵的值将会接近![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo1y1KFJGgRqLZrPQWiccbRtrjsA5mYQXFBUubkibShnyvMHSvdeIeZhicg/0?wx_fmt=png)[^1]。这两个特征在直觉上我们都会觉得它适合做代价函数。事实上，我们的均方代价函数也同时满足这两个特征。这对于交叉熵来说是一个好消息。而且交叉熵有另一个均方代价函数不具备的特征，它能够避免学习速率降低的情况。为了理解这个，我们需要计算一下交叉熵关于权重的偏导
    - 我们权重的学习速率可以被![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPovGD0YNORob35ISCu9mJbiadic8As0aFG9iaN7bibAkqKuaLGbPK2dgnIhQ/0?wx_fmt=png)控制，也就是被输出结果的误差所控制。误差越大我们的神经元学习速率越大。这正是我们直觉上所期待的那样。另外它能避免学习减速，这是![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoQGzSoqvJrPOmKUWCcJnqic8ltE7F09fFzCWiayF7moyD4jpgjpIWKWicw/0?wx_fmt=png)一项导致的。当我们使用交叉熵时，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoQGzSoqvJrPOmKUWCcJnqic8ltE7F09fFzCWiayF7moyD4jpgjpIWKWicw/0?wx_fmt=png)这一项会被抵消掉，因此我们不必担心它会变小。这种消除是交叉熵代价函数背后所带来的惊喜。实际上，这并不是一个惊喜。稍后我们会看到，我们特意选取了具有这种特性的函数。
    - 我需要假设![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo21q1ZibJXb8E4ibAsUTugUCrvWbgnibVibrTAIuhT2kAuwOmQlLWdVNqzg/0?wx_fmt=png)的输出只能为![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPo1y1KFJGgRqLZrPQWiccbRtrjsA5mYQXFBUubkibShnyvMHSvdeIeZhicg/0?wx_fmt=png)或者![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoPHqS23iaepueGe2ttJUPNXr4UKlqmdRGMGyMRoCdK7BpGUO2kjW7OYQ/0?wx_fmt=png)。这种情况特别在分类问题，或者在计算布尔函数时出现。如果你想知道如果我们不做这个假设时会发生什么，请查看本节最后的练习。
    - 代价函数发生改变之后我们不能很精确的定义什么是「相同」的学习速率
    - 你可能会反对学习速率的改变，因为这会让上面的例子变得没有意义。如果我们随意选取学习速率那么谁还会在意神经元学习地有多快呢？这种反对偏离了重点。这个例子的重点不是再说学习速度的绝对值。它是在说明学习速度是如何变化的。当我们使用均方误差代价函数时，如果选取一个错的离谱的开始，那么学习速度会明显降低；而我们使用交叉熵时，这种情况下学习速度并没有降低。这根本不取决于我们的学习速率是如何设定的。
    - 交叉熵用于单个神经元的情况。事实上，这很容易推广到多层神经网络上。我们假设![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoSyV89FS9PiaFm4znPn8LSImgkbatvlU5Sgna6Cd8aUuKtS6rZJnln3g/0?wx_fmt=png)是我们期望的输出，例如，在神经元的最后一层，![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoNHKULcnicFuXuycYYNibdbt7gMkic1kePKxOwic769nPZs2SqH3DuQV8yQ/0?wx_fmt=png)是真实的输出
    - 如果输出神经元是sigmoid神经元的话，交叉熵都是更好的选择
    - **输出层是线性神经元（linear neurons）的时候使用均方误差**
    - 有一个多层神经网络。假设最后一层的所有神经元都是*线性神经元（linear neurons）*意味着我们不用sigmoid作为激活函数，输出仅仅是![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoE4WJorEnqFticIeun4h63icleicAj6cQDCwgFFHZWrO4AqlJkaibFMbnIw/0?wx_fmt=png)。如果我们用均方误差函数时，输出误差![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPoicLKF7kicZsuy1Kl4QDrxZuJ9U9ib1U1RrOGte8ae34aicbSmckcO9XL7w/0?wx_fmt=png)对于每个训练输入![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjpU2ZKG81XIBtUZCbDHiadPokk0JIFO8ogl7ianAnK8NtStP5icZ47yrcp4uKicBh0Magex2UNtqSEibkQ/0?wx_fmt=png)为
    - 前的问题类似，利用这个表达式我们在输出层对权重和偏移求导
  - 使用交叉熵解决手写数字识别问题
    - 我们可以很容易在程序中将交叉熵应用于梯度下降法（gradient descent）和反向传播算法（backpropagation）
    - 先看一下新程序解决手写数字问题的效果如何。和第一章的例子一样，我们的网络将用30个隐层神经元，mini-batch的大小选为10。设定学习速率为η=0.52，迭代次数选为30。
    - 这条命令是用来初始化权重和偏移的。我们需要运行一下这条命令，因为在这章的后面我会改变我们网络初始化权重的默认方式。
    - 接下来我们选取隐藏层神经元个数为100，代价函数为交叉熵函数，其它的参数保持不变。在
    - 如学习速率，mini-batch的大小等等。
  - 交叉熵意味着什么，从哪里来
    -   交叉熵的直观意义是什
    - 在信息论领域是有一种标准方式来解释交叉熵的。大致说来，想法就是：交叉熵是对惊讶的测度
  - Sotfmax
    - 主要使用交叉熵代价函数来解决学习速度衰退的问题。不过，我想首先简要的介绍一下解决这个问题的另一种方法，这种方法是基于神经元中所谓的 *softmax* 层
    - 面因为 softmax 在本质上很有趣，另一方面因为我们将要在第六章使用 softmax 层，那时候我们要讨论深度神经网络。
    - softmax 的主要思想是为我们的神经网络定义一种新的输出层。跟之前的 sigmoid 层一样，它也是对加权输入
    - 不一样的地方是，在获取输出结果的时候我们并不使用 sigmoid 函数。取而代之，在 softmax 层中我们使用 *softmax 函数 *应用到![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjqia4XmnQMJGo1QDw4T2Oib0rdCDkib4j0hyAMkpPbeJicicenJmL1fIzYW2O3RYsL4A1RCrk38Xs9YWnQ/0?wx_fmt=png)。根据这个函数，第![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjqia4XmnQMJGo1QDw4T2Oib0rC6yqxAMwymiaLIfIL4G8ibbMHSGRKGmQ3KibdfTeasyaeyQ2bFvsaHekQ/0?wx_fmt=png)个输出神经元的激活值![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjqia4XmnQMJGo1QDw4T2Oib0rNCae9QOl48g3UgKDx3fia51vzCW5PyJNyQdMMwTzJRtLu6EYF4I8KcQ/0?wx_fmt=png)是
    - ​































### 关于「练手到熟练项目」：1%

Java基础：参考：

##### **Web项目:**

- Python 爬虫：参考：[Scrapy](http://wiki.jikexueyuan.com/project/scrapy/)、


- Java Web：参考：[Java Web](http://wiki.jikexueyuan.com/project/java-web/)、
- PHP+MySQL搭建网页：参考：[PHP+MySQL](http://wiki.jikexueyuan.com/project/php-and-mysql-web/) 、

##### **数据挖掘／自然语言处理项目：**

- 建模

##### **算法基础：**

- 数据结构：参考：[剑指Offer](http://wiki.jikexueyuan.com/project/for-offer/) 、[LeetCode](http://wiki.jikexueyuan.com/project/leetcode-book/)
- 算法：参考：

##### **大数据项目：**

- Spark、Scala

### 关于「论文」：1%

数量：

##### [基于关键词及问题主题的问题相似度计算](http://chuansong.me/n/1906250351625) ：

- 社区问答系统、判断问题相似度、推荐问题的答案、避免重复提问。
- 问题包括：问题主题及问题描述
- KT-CNN模型包括：
  - 关键词抽取：
    - 对输入问题S和T,先预处理在通过此模块抽取S和T的关键词序列Ks和Kt
      - **需要：**得分排序序列、TextRank算法、无向有全图、图排序以及选取关键词、TF-IDF
  - 基于关键词相似相异的问题建模:
    - 利用Ks和Kt间相似相异信息，对S和T建模得到特征向量Fs和Ft
      - **需要**：基于文本间相似及相异信息的CNN模型、词向量表示、GloVe模型、语义匹配：相似矩阵、余弦相似度、皮尔森相关系数、矩阵分解、CNN模型卷积层、CNN模型最大池层、
  - 计算主题相似度:
    - 对问题S和T的主题Ts和Tt计算相似度Sim_topic
      - **需要**：向量表示、皮尔森相关系数
  - 问题相似度计算:
    - 基于问题S和T的特征向量Fs和Ft及主题相似度Sim_topic计算S和T的相似度Sim_q
      - **需要**：线性模型的加权相加

##### [人机对话系统中基于关键词的回复生成技术](http://chuansong.me/n/1873214451926)：

- 早期回复生成技术：基于规则：文本分析规则。／／ 序列决策问题：马尔可夫决策
- 回复生成问题：Seq2Seq深度学习框架：用户消息和回复被建模成两个序列，通过大规模训练数据训练模型学习两个序列间的映射关系。
  - 极大似然估计方法的交叉熵损失函数、最大互信息损失、随机变量
- 使用关键词增强生成回复的相关性
  - Seq2BF（sequence to backward and forward sequences）模型
  - 分为两阶段：根据用户消息在词表中选取和消息具有最大互信息的词作为关键词，根据关键词预测回复的剩余部分。
  - 简要介绍：Seq2Seq 框架下基本的回复生成模型：
    - 两部分之encoder：负责将消息编码成一个模型的内部表示
    - 两部分之decoder：在encoder输出条件下，从特定字符开始，逐字生成完整的回复内容。

##### [文本生成概述](http://chuansong.me/n/1851499951313)：

- 根据格式化数据或自然语言文本生成新闻等可解释文本、定义、任务、评价指标、实现方法、数据驱动方法
- 文本生成定义：接受非语言形式的信息作为输入，生成刻度的文字表示。数据到文本的生成。
- 文本生成任务：文本到文本、数据到文本、图像到文本
  - 文本到文本：
    - 文本摘要：
      - 抽取式摘要：信息抽取和规划等主要步骤
        - 主题模型、聚类、SVR(Support Vector Regression)、线性回归、抽取名词短语、动词短语、随机森林、
      - 生成式摘要
- 文本生成方法：基于规则、基于规划、数据驱动
  - 基于语言模型的自然语言生成：n-gram
  - 使用深度学习的自然语言生成

##### [事件演化的规律和模型](http://chuansong.me/n/1835019551922)：

- 事件图谱、图结构、马尔可夫逻辑网络（无向图）、贝叶斯网络（有向五环图）。事理图谱（有向有环图）。
- 事理图谱的定义：
  - 事件：用抽象、泛化的谓词短语来表示。：不关注时间、地点等
  - 事件间顺承关系：两个时间在时间上先后发生的关系。
  - 事件因果关系：满足顺承关系时序约束的基础上，两个事件件有很强的因果性，强调前因后果。
  - 事理图谱**Event Evolutionary Graph** ：描述事件之间顺承、因果关系的事理烟花逻辑有向图。以某个事件节点进行广度优先搜索，扩展得到事件演化链条。
    - 事理图谱是一种概率有向图。概率图模型的贝叶斯网络、马尔可夫逻辑网络、贝叶斯用有向无环图表达变量节点间的条件依赖与独立性关系。马尔可夫随机场采用无向图表达变量间的相互作用关系。

##### [阿里自然语言处理部总监：NLP技术的应用及思考](http://chuansong.me/n/1810576651425) ：

郎君博士，哈工大社会计算与信息检索研究中心博士毕业生，目前为阿里巴巴iDST自然语言处理部总监

- 计算平台、业务层、NLP4大经典AI完全难题：问答、复述、文摘、翻译。
- 阿里需要：技术体系及服务、核心业务快速增长、商业机会
  - 内容搜索、内容推荐、评价、问答、文摘、文本理解
  - 商品搜索、推荐、智能交互、翻译、广告、风控、舆情监控、
- 词法分析：分词、词性、实体：
  - Bi-LSTM-CRF、多领域词表
  - 推荐算法、蚂蚁金服、资讯搜索
- 句法分析：依存句法分析、成分句法分析：
  - Shift-reduce、graph-based、Bi-lSTM
  - 资讯搜索、评价情感分析、商品标题、搜索Query
- 情感分析：情感对象、属性、属性关联
  - 情感辞典挖掘、属性级、句子级、篇章级情感分析
  - 商品评价、问答、品牌舆情、互联网舆情
- 句子生成：句子可控改写、句子压缩
  - Beam Search、 Seq2Seq+Attention
  - 商品标题压缩、资讯标题改写、PUSH消息改写
- 句子相似度：浅层相似度、语义相似度
  - Edit Distance、Word2Vec、DSSM
  - 相似问题、商品重发检测
- 文本分类／聚类：垃圾防控、信息聚合
  - ME, SVM, FastText
  - 商品类目预测、问答意图分析、文本垃圾过滤、舆情聚类、名片OCR后语义识别
- 文本表示：词向量、句子向量、篇章向量、Seq2Seq
  - Word2Vec、LSTM、DSSM、Seq2Seq
- 知识库：电商同义词／上下位、通用同义词／上下位。领域词库、情感词库
  - bootstrapping、click-through mining、word2vec、k-means、CRF
  - 语义归一、语义扩展、Query理解、意图理解、情感分析
- 语料库：分词、词性标注数据、依存句法标注数据
- 标题分析：分词、实体打标、热度计算、中心识别
- 评价系统：treelink模型、maxent模型、贝叶斯模型、dbn模型总体融合
- 决策购买问题：产品化：分为四类：
  - 无效问题过滤：
    - 分类采用LR+GBDT：定制特征
  - 相似问题识别：
    - Doc2Vec 计算相似度、人工评测
  - 页面问答排序
    - 内容丰富度、点赞数、过滤词表匹配数等加权求和、CTR提升
  - 智能分发

##### [基于深度多任务学习的自然语言处理技术](http://chuansong.me/n/1768183851425) 

- 统计自然语言处理依赖标注数据
- 多任务学习Multi-task Learning：
  - 有监督学习Supervised Learning， 利用人工标注的训练数据进行学习
  - 归纳迁移机制，基本目标是提高泛化性能。利用并行训练的方法学习多个任务。 多任务学习的基本假设是多个任务之间具有相关性。
  - 词法、句法、语义分析等多任务，之间存在紧密的内在联系
- 深度多任务学习：
  - 深度学习：建立在含有多层非线性变换的神经网络结构之上，对数据进行抽象表示和学习的一系列机器学习算法。

##### [对话系统的Goal Oriented和Task Oriented 概念的异同](http://chuansong.me/n/1758705951350)

- 人机对话系统：
  - 开放域对话系统：闲聊：微软小冰。 百度问答：度秘。
  - 任务型对话系统：设备控制：Siri。公交线路查询、餐厅预订。

##### [自然语言处理中的知识获取](http://chuansong.me/n/1724644951614)

- 各个行业：教育、医疗、法律等知识服务型行业
- 自然语言中任何问题抽象为：如何从形式与意义的多对多映射中，根据语境选择一种正确的映射。
- 自然语言处理中知识获取的三要素：
  - 显性知识：
    - 元知识，如WordNet、知识图谱
  - 数据：
    - 带标注数据、无标注
  - 学习算法：
    - SVM、CRF等浅层学习模型，人工定义的特征模版抽取特征及特征组合，结合
    - RNN、CNN等深度学习模型，自动学习有效特征及特征组合的能力
- 大数据和深度学习相互依赖的：
  - 一方面大数据需要复杂的学习模型。长尾数据、复杂模型、大数据
  - 另一方面深度学习需要大数据。神经网络机器翻译（NMT）已经迅速超越统计机器翻译（SMT）、端到端
    - 信息抽取两种做法：
      - 1先做句法分析，再做信息抽取
      - 2直接信息抽取，也就是“端到端”，也是分层的

##### [迁移学习：基本概念到相关研究](http://chuansong.me/n/1716697451319)

- 什么是迁移学习：

  - 机器学习的监督学习场景中，如果针对任务和域A训练一个模型，我们假设被提供了此标签数据。
  - 迁移学习借用已存在的相关任务或域的标签数据来处理场景。尝试把源域中解决源任务时获得的知识存储下来，应用到目标域的目标任务中。

- 为什么需要迁移学习：

  - Andrew Ng：迁移学习将会是继监督学习之后的下一个机器学习商业成功驱动力。
  - ml在产业界的应用和成功主要：监督学习的驱动。
  - 无监督学习是实现通用人工智能的关键成分。
  - 产业界对ml的应用分2类：
    - 一方面：先进模型
    - 另一方面：大量标签数据使得模型成功
  - 迁移学习可以帮助我们处理全新的场景，是ml在没有大量标签任务和域中规模化应用所必须的。

- 迁移学习定义

  - 给定一个源域 Ds，一个对应的源任务 Ts，还有目标域 Dt，以及目标任务 Tt，现在，迁移学习的目的就是：在 Ds≠Dt，Ts≠Tt 的情况下，让我们在具备来源于 Ds 和 Ts 的信息时，学习得到目标域 Dt 中的条件概率分布 P（Yt|Xt）

- 迁移学习场景

  - 给定源域和目标域 Ds 和 Dt，其中，D={X,P(X)}，并且给定源任务和目标任务 Ts 和 Tt，其中 T={Y,P(Y|X)}。源和目标的情况可以以四种方式变化

- 迁移学习的应用

  - 从模拟中学习：google自动驾驶、机器人等

- 迁移学习方法

  - 使用预训练CNN特征
  - 理解卷积神经网络、全连接层


##### [如何让人工智能学会用数据说话](http://chuansong.me/n/1668814251914)

- 基于结构化数据的的文本生成
  - 机器翻译、文本摘要、诗词生成等都属于文本生成的范畴：
    - 共同点：用户输入非结构化文本，机器根据目标输出相应文本
- 结构化文本生成：特点：基于数据和事实
- 文本生成的典型商业应用：财经、体育类新闻报道的生成、产品描述、商业数据的分析和解释、物联网数据的分析。比如：天气预报自动生成。
- 文本生成的技术发展：
  - 对选定的数据记录，用自然语言描述出来
- 方法：
  - 早期：基于规则：三个独立的模块：
    - 内容规划(Content planning), 即选择描述那些数据记录或数据域
    - 句子规划(Sentence planning), 即决定所选择的数据记录或数据域在句子中的顺序
    - 句子实现(Surface realization), 即基于句子规划生成实际的文本。
  - 基于神经网络的方法：
    - 基于神经语言模型(Neural Language Model)
    - 基于神经机器翻译(Neural Machine Translation)
    - Semantic Controlled LSTM（Long Short-term Memory）模型:用于文本生成：在LSTM基础上引入了控制门读取结构化数据信息。
- 数据：
  - 天气预报、维基百科人物传记、基于对话的人机对话数据集

##### [卷积神经网络在句子分类上的应用](http://chuansong.me/n/1723633)

- 基于预先训练的词向量而训练的卷积神经网络(CNN-Convolutional Neural Networks)、文本分类、语义分析、词向量word vectors 
- 卷积神经网络CNN利用一个卷积层进行特征提取，最初CV，后来在nlp也得到应用，如语义分析、搜索短语检索、句子分类
- 特征向量使用max-over-time池化、word2vec词向量、

##### [自然语言中的Attention Model](http://chuansong.me/n/2215468)

- 先说Encoder-Decoder框架
  - AM模型基本附着在Encoder-Decoder框架下的。但本身并不依赖于Encoder-Decoder模型：适合处理由一个句子生成另一个句子的通用处理模型。
  - 具体使用什么模型自己定：CNN/RNN/BiRNN/GRU/LSTM/Deep LSTM
- AM：以上的还没有体现出“注意力模型”。

##### [深度学习：推动NLP领域发展的新引擎](http://chuansong.me/n/2793709)

- Word Embedding: word2vec能把词变成向量：忽略词之间的关系(句法关系)、词的顺序等
  - 引入词的关系：Dependency Parser：把抽取出来的Relation作为词的Context
  - 改进Bag of Words:
  - 外部资源和知识库：word2vec只用了词的上下文共现， 没有用外部资源如词典知识库
- RNN／LSTM／CNN
  - RNN相关的模型如LSTM基本上算法是解决结构化问题的标准。比普通的FeesForward Network, RNN 有记忆能力
  - 普通的神经网络只会在学习的时候“记忆”，也就是反向传播算法学习参数。然后不“记忆”了。训练好之后，不管什么时候来一个相同的输入，输出一样。对于image classification没问题，但是speech recognition等nlptask，数据有时序或者结构的。
  - RNN具有“记忆”能力。前一个输出会影响后面的判断。比如：前一个He,后面出现is的概率比are高很多。
  - 最简单RNN直接把前一个时间点的输出作为当前出入的一部分，有梯度消失的问题。比较流行的改进如LSTM和GRU等模型通过gate开关，判断是否需要遗忘／记忆之前的状态，以及当前状态是否需要输出到下个时间点。比如语言模型。
  - CNN最早图像。通过卷积发现位置无关的feature，而且这些feature的参数相同，
    - machine translation、语义角色标注Sematic Role labeling
- Multi-model deep learning
- Reasoning, Attention and Memory
  - RNN/LSTM是模拟人类大脑的记忆机制，但除了记忆之外，Attention也是非常有用的机制

##### [深度学习浪潮中的自然语言处理技术](http://chuansong.me/n/472336251048)

- 目前，机器学习技术为自然语言的歧义、动态性等提供了可行的解决方案，成为研究主流，称为统计自然语言处理。
- 一个统计自然语言处理系统通常由2部分构成：
  - 训练数据（样本）
  - 统计模型（算法）
- 但是传统的机器学习方法在数据获取和模型构建等方面存在严重问题：
  - 大规模标注数据难以获得，带来了严重的数据稀疏问题。
  - 需要人工设计模型所需的特征及特征组合。需要深刻理解和丰富经验
- 基于深度学习的自然语言处理
  - 建立在含有多层非线性变换的神将网络结构之上，对数据的表示进行抽象和学习的一系列机器学习算法。
  - 深度学习为自然语言处理的研究主要带来两方面变化：
    - 一方面使用统一的分布式(低维、稠密、连续)向量表示不同粒度的语言单元，如词、短语、篇章
    - 一方面使用循环、卷积、递归等神经网络模型对不同的语言单元向量进行组合，获得更大语言单元的表示
  - 分布式表示：
    - 深度学习最早在nlp的应用是神将网络语言模型：基本假设低维、稠密、连续的向量表示词汇，又称为分布式词表示(Distributed Word Representation)或词嵌入(Word Embedding)。 可以将相似的词汇表示为相似的向量。
    - 理论上，将原有高维、稀疏、离散的词汇表示方法（One-hot表示）映射为分布式表示是一种降维方法，可有效克服机器学习的维度灾难（Curse of Dimensionality）问题，从而获得更好的学习效果。
  - 语义组合(Semantic Composition)
    - 分布式词表示的思想可以进一步扩展，可通过组合（Composition）的方式来表示短语、句子甚至是篇章等更大粒度的语言单元。
    - 三种神经网络结构实现不同的组合方式
      - 循环神经网络（顺序组合）RNN，Recurrent Neural Network
        - 从左至右顺序的对句子中的单元进行两两组合：“我”和“喜欢”组合，生成隐层h1。将h1和“红”组合，生成h2.类推。传统的RNN存在严重梯度消失（Vanishing Gradient）或者梯度爆炸（Exploding Gradient）问题。
        - 深度学习中一些常用的技术，如使用ReLU激活函数、正则化以及恰当的初始化权重参数等都可以部分解决这一问题
        - 另一类更好的解决方案是减小网络的层数，以LSTM和GRU等为代表的带门循环神经网络（Gated RNN）都是这种思路，即通过对网络中门的控制，来强调或忘记某些输入，从而缩短了其前序输入到输出的网络层数，从而减小了由于层数较多而引起的梯度消失或者爆炸问题
      - 卷积神经网络（局部组合）CNN，Convolutional Neural Network
        - 隐含层神经元只与部分输入层神经元连接，同时不同隐含层神经元的局部连接权值共享。如评论文本分类，最终褒贬性由局部短语决定，且与顺序无关。
        - 由于存在局部接收域性质，各个隐含神经元的计算可以并行的进行，这就可以充分利用现代的硬件设备（如GPU），加速卷积神经网络的计算，这一点在循环神经网络中是较难实现的
      - 递归神经网络（句法结构组合）RecNN，Recursive Neural Network
        - 首先对句子进行语法分析，将结构转化为树状结构，构建深度神经网络。
  - 很多自然语言任务如对话生成，有赖于更大的上下文或语境，传统的基于人工定义特征的方式很难对其进行建模，深度学习模型则提供了一种对语境进行建模的有效方式 
  - 无论何种神经网络模型，都是基于固定的网络结构进行组合，传统的有监督学习框架很难实现该目标，而强化学习（Reinforcement Learning）框架为我们提供了一种自动学习动态网络结构的途径。

[基于深度学习的关系抽取](http://chuansong.me/n/831970551568)

- 信息抽取旨在从大规模非结构或半结构的自然语言文本中抽取结构化信息。关系抽取是其中的重要子任务之一，主要目的是从文本中识别实体并抽取实体之间的语义关系。
- 现有主流的关系抽取技术分为有监督的学习方法、半监督的学习方法和无监督的学习方法三种
  - 有监督的学习方法将关系抽取任务当做分类问题，根据训练数据设计有效的特征，从而学习各种分类模型，然后使用训练好的分类器预测关系。该方法的问题在于需要大量的人工标注训练语料，而语料标注工作通常非常耗时耗力
  - 半监督的学习方法主要采用Bootstrapping进行关系抽取。对于要抽取的关系，该方法首先手工设定若干种子实例，然后迭代地从数据从抽取关系对应的关系模板和更多的实例
  - 无监督学习方法假设拥有相同语义关系的实体对拥有相似的上下文信息。因此可以利用每个实体对对应上下文信息来代表该实体对的语义关系，并对所有实体对的语义关系进行聚类
  - 有监督的学习方法能够抽取更有效的特征，其准确率和召回率都更高。因此有监督的学习方法受到了越来越多学者的关注
- 基于有监督学习的关系抽取
  - 需要大量人工标注的训练数据，从村里安数据中自动学习关系对应的抽取模式。
  - 远程监督
- 基于深度学习的关系抽取
  - 有监督的依赖标注等分类特征，并且存在错误
  - 递归神经网络解决关系抽取问题
  - 卷积神经网络关系抽取
  - 基于端到端神经网络的关系抽取模型：双向LSTM(Long-Short-Term-Memory)
- 总结及未来趋势
  - 基于句法树的树形LSTM神经网络模型在关系抽取上取得了不错的效果
  - 目前的神经网络关系抽取主要用于预先设定好的关系集合。而面向开放领域的关系抽取，仍然是基于模板等比较传统的方法


##### [基于深度学习的依存句法分析](http://chuansong.me/n/710400151780)  

- 句法分析：句子从词语的序列形式按照语法体系转化为图结构（树结构）。以刻画句子内部的句法关系（主谓宾）。使用依存户连接句子中两个具有一定句法关系的词语，最终形成一颗句法依存树。
- 主流依存句法分析方法：
  - 基于图（Graph-based）
    - 将依存句法分析看成从完全有向图中寻找最大生成树的问题，图的两边表示两个词之间存在某种句法关系的可能性
  - 基于转移（Transition-based）
    - 通过规约等转移动作构建一棵依存句法树，学习目标是寻找最优动作序列
- 深度学习技术：
  - 建立在含有多层非线性变换的神经网络结构之上，对数据抽象和学习的一系列机器学习算法。
- 基于转移的依存句法分析方法：
  - 一系列由初始到终止的状态（State或Configuration）表示句法分析的过程。一个状态由栈（Stack）、缓存（Buffer)以及部分分析好的依存弧构成。栈存储已经分析的词，缓存表示待分析的词
  - 学习一个分类器，输入状态，输出该状态下最可能动作：贪心解码算法
  - 抽取出特征后，传统方法：采用线性分类器，即将特征进行线性加权组合，结合系数为分类器学习获得的权重，选取分数最高的类别作为采取的动作
- 基于深度学习的依存句法分析：
  - 贪心解码算法：先从一个状态提取一些重要核心特征，与传统高维、稀疏、离散向量One-hot表示不同，使用低维、稠密、连续的分布式向量来表示特征。
    - 相似的词可用相似的向量表示：克服数据稀疏问题
    - 分布式表示是一种降维方法，克服维度灾难（Curse of Dimensionality）
  - 全局解码算法：
    - 用柱搜索（Beam Search）等近似全局搜索／解码算法总和考虑多个状态之间的依赖关系。
- 目前还需要利用传统方法构造转移系统，同时全局解码算法有助于系统性能的进步提升
  - 序列到序列（Sequence to Sequence），也称编码解码（EncoderDecoder）方法在多个自然语言处理任务中广泛应用：机器翻译、阅读理解等。
  - 依存句法是典型的结构话学习问题，在nlp中，很多结构话学习任务：分词、词性标注等，都可以借鉴上述基于转移的算法框架解决。
  - 很多上层应用依赖于句法分析。尤其是深度学习中递归神经网络方法就是依赖句法分析结果进行语义的递归组合。

##### [深度学习的五个挑战和其解决方案](http://chuansong.me/n/1664863851518)

- 机器学习的各个主要方向，从底层的深度学习分布式机器学习平台(AI的Infrastructure)到中层的深度学习、强化学习、符号学习算法以及再上面的机器学习理论。
- 新的基于CNN的深度模型叫做残差网络，这个残差网络深度高达152层，取得了当时图象识别比赛上面最好的成绩
- 深度学习里最经典的模型：
  - 全连接的神经网络，就是每相临的两层之间节点之间是通过边全连接；
  - 再就是卷积神经网络，这个在计算机视觉里面用得非常多；
  - 再就是循环神经网络RNN，这个在对系列进行建模，例如自然语言处理或者语音信号里面用得很多，这些都是非常成功的深度神经网络的模型。
  - 还有一个非常重要的技术就是深度强化学习技术，这是深度学习和强化学习的结合，也是AlphaGo系统所采用的技术
- 当前深度学习的一个前沿就是如何从无标注的数据里面进行学习。现在已经有相关的研究工作，包括最近比较火的生成式对抗网络，以及我们自己提出的对偶学习。
  - 生成式对抗网络的主要目的是学到一个生成模型
  - 它是同时学习两个神经网络：一个神经网络生成图像，另外一个神经网络给图像进行分类，区分真实的图像和生成的图像。在生成式对抗网络里面，第一个神经网络也就是生成式神经网络，它的目的是希望生成的图像非常像自然界的真实图像，这样的话，那后面的第二个网络，也就是那个分类器没办法区分真实世界的图像和生成的图像；而第二个神经网络，也就是分类器，它的目的是希望能够正确的把生成的图像也就是假的图像和真实的自然界图像能够区分开。大家可以看到，这两个神经网络的目的其实是不一样的，他们一起进行训练，就可以得到一个很好的生成式神经网络
  - 针对如何从无标注的数据进行学习，我们组里面提出了一个新思路，叫做对偶学习。对偶学习的思路和前面生成式对抗学习会非常不一样。对偶学习的提出是受到一个现象的启发：我们发现很多人工智能的任务在结构上有对偶属性
    - 搜索引擎最主要的任务是针对用户提交的检索词匹配一些文档，返回最相关的文档；当广告商提交一个广告之后，广告平台需要给他推荐一些关健词使得他的广告在用户搜索这些词能够展现出来被用户点击
- 深度学习面临的第二个挑战就是如何把大模型变成小模型
  - CNN模型，也就是卷积神经网络，做模型压缩；
    - 剪枝：边权重小的去掉
    - 权值共享：上百万权值聚类，用均值代替这一类权值
    - 量化：降低浮点型精度
    - 二进制神经网络：原来32bit权值现在1bit
  - 针对一些序列模型或者类似自然语言处理的RNN模型如何做一个更巧妙的算法，使得它模型变小，并且同时精度没有损失。
    - 新的循环神经网络LightRNN：不是模型压缩降低模型大小，而是算法
    - 每个词要做词嵌入（word embedding）语义相似或相近的词在向量空间里的向量也比较接近，这样可以表达词之间语义信息或相似性。
    - 我们不用一个向量表示一个词，而是两个向量表达一个词。行／列
- 今年的人工智能国际顶级会议AAAI 2017的最佳论文奖，颁给了一个利用物理或者是一些领域的专业知识来帮助深度神经网络做无标注数据学习的项目。论文里的具体例子是上面这张图里面一个人扔枕头的过程，论文想解决的问题是从视频里检测这个枕头，并且跟踪这个枕头的运动轨迹。如果我们没有一些领域的知识，就需要大量的人工标注的数据，比如说把枕头标注出来，每帧图像的哪块区域是枕头，它的轨迹是什么样子的。实际上因为我们知道，枕头的运动轨迹应该是抛物线，二次型，结合这种物理知识，我们就不需要标注的数据，能够把这个枕头给检测出来，并且把它的轨迹准确的预测出来。这篇论文之所以获得了最佳论文奖，也是因为它把知识和数据结合起来，实现了从无标注数据进行学习的可能
- AI技术来分析股票：动态决策性问题：难点：时变性
- 决策第二点：各种因素相互影响：静态认知性任务我们的预测结果不会对问题

##### [理解LSTM网络](http://chuansong.me/n/1756021)

- Recurrent Neural Networks
  - 传统神经网络不能处理的问题：使用先前的事件推断后续的事件
  - RNN解决了这个问题：包含循环的网络：允许信息的持久化
  - 同一神经网络的多次复制，每个神经网络模块把消息传递给下一个
  - 链式特征揭示了RNN本质上与序列和列表相关的
  - RNN的语音识别、建模、翻译、图片描述等
  - 应用成功的关键之处：LSTM的使用，是一种特别的RNN，比标准的RNN在很多任务都表现的更好。
- 长期依赖问题：
  - RNN的关键：用来连接先前的信息到当前的任务上，还有很多依赖因素
  - 有时候，我们仅仅需要知道先前的信息来执行当前的任务。例如：语言模型基于当前词语来预测下一个词。如果预测“the clouds are in the sky”最后的词，我们并不需要任何其他的上下文--因此下一个词应该是sky。RNN可以学会使用先前的信息。
  - 但是更加复杂的场景下：假设我们试着去预测“I grew up in France… I speak fluent French”最后的词，相关信息和当前预测位置之间的间隔变得相当大。当间隔大的时候，RNN会丧失学习到连接如此远的信息的能力。
- LSTM网络-Long Short Term 网络
  - RNN特殊的类型。可以学习长期依赖
  - 通过刻意的设计避免长期依赖问题。记住长期的信息在实践中是LSTM的默认行为
  - 所有RNN都具有一种重复神经网络模块的链式形式，标准RNN中，重复的模块只有一个简单结构：例如tanh层。
  - LSTM同样如此：但是重复的模块有不同的结构
- LSTM的核心思想：
  - 关键：细胞状态：水平线在图上方贯穿运行。
  - LSTM有通过精心设计的称为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。包含一个sigmoid神经网络层和一个pointwise 乘法操作。Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”！
- 逐步理解LSTM：
  - 第一步决定丢弃什么信息。通过忘记门层完成。
  - 下一步决定什么样的新信息被存在细胞状态中。
    - 第一，sigmoid层称“输入门层”决定什么值我们将要更新。
    - 根据信息产生对状态的更新
    - 最后确定输出什么值。这个输出基于我们的细胞状态。
- LSTM的变体
  - GRU

##### [浅谈基于张量分解的隐变量模型参数学习](http://chuansong.me/n/1805856)

- 很多工作使用张量分解学习隐变量模型的参数，可以获得参数的全局最优解
- 隐变量模型本质上一个定义在两类变量上的概率分布：隐含变量和观测变量：隐变量模型的两个最基本任务：
  - 给定观测变量，推断隐含变量，在概率图模型：infernce
  - 估计模型中参数：learning
- 学习概率分布的参数：通常方法；最大似然估计、矩估计。
  - 隐变量模型的参数学习使用最多的是最大似然估计：给定数据，写出似然函数，优化似然函数，估计参数。：参数估计的主流方法。 缺点：隐变量模型的似然函数基本上都是非凸函数，很难获得全局最优解。优化似然函数的常用方法：EM、variational EM通常只能获得局部最优解。
  - 为了获得参数的全局最优解：矩估计：基于张量分解的方法就属于此类：基本思想：求样本的低阶矩，通过解方程得到参数的值。求解一阶矩和二阶矩。但是方程不一定能写出来。张量方法目前只适用于某些隐变量模型，隐马尔可夫模型，话题模型，高斯混合模型，并不像最大似然方法普遍适用于任何模型
  - 张量分解本质上：优化问题：非凸优化问题。如果这个非凸优化中得到的是局部最优解。但是如果张量是对称的。尽管非凸优化，也能获取全局最优解。

##### [人工智能之争](http://chuansong.me/n/1792248)

- 传统人工智能：自然语言翻译、符号推理（symbolicreasoning）、博弈论（game playing）等问题：专家系统：运用规则记录专家的经验：医生诊断经验的模型
- 人机交互human computer interaction：图形用户界面
- 机器学习：数理统计工具开发不同的识别和分类算法:识别目标、发现数据中模式、用于机器人的策略

##### [Facebook人工智能研究最新进展](http://chuansong.me/n/1897593)

- 最大挑战：无监督学习
  - 因为人类和动物使用对多的就是这种学习方式

##### [知识图谱的应用](http://chuansong.me/n/1999287)

- 什么是知识图谱
  - 本质：语义网络：基于图的数据结构：节点（point）和边（edge）组成
  - 每个节点：表示现实的实体。每条边：实体与实体之间的“关系”
  - 知识图谱：关系的最有效的表示方式，把所有不同种类的信息连接在一起的到的关系网络，知识图谱提供了从“关系”角度去分析问题的能力。
  - 最初：用来优化现有的搜索引擎，不同于关键词搜索的传统搜索引擎，知识图谱可用来更好的查询复杂的官联系信息。
- 知识图谱的表示
  - 描述：事实。
  - 属性图和传统的RDF格式都可以作为知识图谱的表示和存储方式
- 知识图谱的存储
  - 知识图谱是基于图的数据结构。存储方式主要有两种：
    - RDF存储
    - 图数据库
  - 如果需要设计的知识图谱简单，查询不会涉及到1度以上的关联查询。可以选择用关系型数据存储格式来保存知识图谱。对于复杂的关系网络知识图谱的的优点还是明显的。
  - 把实体和关系存储在图数据结构是一种复合整个故事逻辑的好方式
- 应用
  - 反欺诈：风控：基于大数据的反欺诈难点：不同来源的数据（结构化、非结构化）整合在一起，构建反欺诈引擎，并有效识别出欺诈。涉及到复杂的关系网络。知识图谱，作为关系的直接表示方式，可以很好的解决这两个问题。
  - 构建多数据源的知识图谱
  - 不一致性验证设计到知识推理。可以理解成链路预测，也就是从已有的关系图谱里推导出新的关系或连接。
  - 异常分析：基于图
    - 静态分析：给定图结构和某个点，从中发现一些异常点
    - 动态分析：分析其结构随时间变化的趋势
  - 除了贷前的风险控制，也可以在贷后发挥其强大的作用。
  - 智能搜索
  - 精准营销：结合多种数据源分析实体之间的关系。对用户的行为有更好的理解。
- 数据的噪声
  - 数据本身错误：不一致性验证
  - 数据冗余：涉及“消歧分析”
- 非结构化数据处理能力
  - 数据挖掘、nlp、ml
- 知识推理
  - 人类智能的重要特征：需要规则的支持。常用的推理算法：基于逻辑的推理（Logic）和基于分布式表示方法(Distributed Representation)的推理
- 大数据、小样本、构建有效的生态闭环是关键
  - 面临的依然是小样本问题，也就是样本数量少。实际上，我们能拿到的欺诈样本数量不多，即便有几百万个贷款申请，最后被我们标记为欺诈的样本很可能也就几万个而已
  - 所谓的生态闭环，指的是构建有效的自反馈系统使其能够实时地反馈给我们的模型，并使得模型不断地自优化从而提升准确率。

##### [AlphaGo原理解析](http://chuansong.me/n/2658183)

- 围棋棋盘是19x19路，所以一共是361个交叉点，每个交叉点有三种状态，可以用1表示黑子，-1表示白字，0表示无子，考虑到每个位置还可能有落子的时间、这个位置的气等其他信息，我们可以用一个361 * n维的向量来表示一个棋盘的状态。我们把一个棋盘状态向量记为s
- 当状态s下，我们暂时不考虑无法落子的地方，可供下一步落子的空间也是361个。我们把下一步的落子的行动也用361维的向量来表示，记为a。
- 这样，设计一个围棋人工智能的程序，就转换成为了，任意给定一个s状态，寻找最好的应对策略a，让你的程序按照这个策略走，最后获得棋盘上最大的地盘
- 第一招：深度卷积神经网络：
- 第二：MCTS：蒙塔卡洛搜索树Monte-Carlo Tree Search：
  - 没有任何人工的feature，完全依靠规则本身。靠一种类似遗传算法的自我进化
  - 可以连续运行
- 第三：自我进化，强化学习

[神经网络做唐诗](http://chuansong.me/n/2247902)

- [github地址](https://github.com/GeekQi/tangshi-rnn)
- 神将网络在cv和nlp方面效果非常好，原因有如下：
  - 传统的机器学习建立在统计基础上，但是当数据与数据之间的关系难以用统计描述，传统方法不行
  - 传统ml需要专家知识挑选特征。特征好坏与学习成果有很大关系。
- 神经网络的厉害之处就是克服了以上两点：
  - 一是每个神经元都有非线性的公式，可以得到复杂的难以用数学公式描述的关系
  - 二是语音识别的网络，训练时都是原始数据输入，到结构输出模型（End2End）
- 简单的神经网络有局限：RNN递归神经网络（Recurrent Neural Network）本身限制
  - 每次的输出作为下一次的输入返回到神经网络中。训练神经网络就是把这个过程逆向，从最后的输出开始往之前的输出推，每一次推的时候要用到当时的输入和输出计算梯度。
    - 当把第1000个字符的梯度退回到第1个时间状态有两个问题：
      - 存储999个中间值，吃光内存
      - Vanishing/Exploding Gradient, 太远的gradient可能就太大爆掉，或者太小对神经网络完全没有影响
- 但是近体诗：五言和七言的比较适合这个神经网络
  - 字数少、句式固定、规律明确。
    - 大概$150的AWS的GPU 节点使用费，和一些其他忽略不计的数据处理的各种CPU节点。
    - 一周收集并清理全唐诗。
    - 然后训练的时候走了很多弯路，最后这个model在一个月之后才弄出来的。
    - 处理平仄花了一周。
- LSTM和RNN的对比
  - RNN有两个著名的衍生姐妹，Gated Recurrent Unit和Long-Short-Term Memory。
  - LSTM训练时间长，最后效果好不少
- Word Embedding重要性及词典大小
  - 神经网络每个输入输出都是向量或矩阵。为了能把所有字都转化成向量，用到的技巧就是embedding：每个字变成独特的向量。里面只有有一个1
  - 神经网络的第一层实现功能就是把这个很长的one hot vector 变成一个独特的长度更低的向量
  - 实际使用取了频率最高的2000个字，用34%的独立字符覆盖94%的总体字数
- 加入起始和终止字符，让结构更加明确。^ 表示开始。 $表示结束
- 利用Dropout增加准确性
  - 训练的时候一定数量内部的神经元被随机设置为0，但是训练完了又回复原状。也就是遮挡住一部分特征。比如小孩识别汽车，每次看一部分，但是给张完整的，也能看出是汽车
- 不同SGD方法效果不明显
- 生成字符表是一个概率分布
  - 这个神经网络的生成结构：每次在做预测的时候，并不是给出某一个字符，而是给出整个词典里所有字可能出现的概率。
- 将平水韵作为神经网络的输出过滤
  - 关键是押韵效果不好，需要平仄的地方进行过滤


##### [神经网络十大误解分析](http://chuansong.me/n/329467651436)

- 机器学习中最流行最强大的一类
  - 计量金融中，常被用作时间序列预测、构建专用指标、算法交易、证券分类、信用风险建模等，但是不可靠
- 神经网络更接近曲线拟合（curve fitting）和回归分析（regression analysis）等统计方法。曲线拟合即函数逼近，逼近复杂的数学函数
- 神经网络由互连节点层组成。单个节点称为感知器（perceptron），类似与一个多元线性回归（multiple linear regression）。
  - 多元线性回归和感知器之间的不同之处在于：感知器将多元线性回归生成的信号送给可能线性可能非线性的激活函数中。在多层感知器（MLP）中，感知器按层级排布，层层之间互相连接。
  - MLP中有三种类型的层：输入层、隐藏层、输出层。
    - 输入层接收输入模式而输出层可以包含一个分类列表
    - 隐藏层调整输入的权重，直到将NN的误差降低到最小（另一种解释：隐藏层提取输入数据中的显著特征，这些特征有关于输出的预测能力）
  - 映射输入：输出
    - 感知器接收输入向量。通过感知器的权重向量进行加权。在多元线性回归的背景中，这些可被认为是回归系数或beta系数。使用该总和产物得到净值的神经元被称为求和单元（summation unit）
    - 净输入信号减去偏差theta后被输入一些激活函数f()。激活函数通常是单调递增函数，其值位于 (0,1) 或 (-1,1) 之间，激活函数可以是线性的，也可以是非线性的
    - 最简单的神经网络只有一个映射输入到输出的神经元。对于给定模式 p，该网络的目标是相对一些给定的训练模式 tp 的一些一只的目标值来最小化输出信号 op 的误差。比如，如果该神经元应该映射 p 到 -1，但却将其映射到了 1，那么，根据距离的求和平方测定，神经元的误差为 4，即 (-1-1)^2
  - 分层：
    - 感知器被分层进行组织：第一层：输入层，接受训练集。隐藏层将前一层的输出作为下一层的输入，下一层的输出又作为另一层的输入。隐藏层：提取输入数据中的显著特征。这些特征可以预测输出。这个过程称为特征提取（feature extraction），而且和主成分分析（PCA）等统计技术相似功能。
  - 学习规则：
    - 神经网络的目标是最小化一些错误度量（measure of error）,最常见的错误度量是平方误差和（Sum squared error(SSE)） ε
    - 鉴于该网络的目标是最小化 ε，可以使用优化算法调整该NN中的权重。最常见学习算法是梯度下降法：计算相对于NN中每一层的权重的误差偏导数，然后在提督相反的方向上移动
- NN的不同框架
  - 多层感知器：最简单的NN结构。任何NN的性能，是其结构和权重的一个函数。
  - RNN（递归NN）：一些或所有的连接倒流，意味着反馈环路存在于NN中。
  - Boltzmann NN 最早的全连接NN之一，也就是Boltzmann机。：Hopfield递归NN的蒙特卡洛版。
  - 深度NN最大问题之一：尤其是不稳定金融市场环境下，是过度拟合
  - 自适应NN（AdaptiveNN）能够在学习中同时自适应、并优化自身结构的NN。
  - 径向基函数网络：利用径向基函数作为激活功能。常用的函数为高斯分布。还用于支持向量机的内核
- 如果NN太大／太小，NN会出现过拟合／拟合不够，也就是网络无法顺利泛化样本
- 利用相关性去选择输入变量有两个问题：如果使用线性相关矩阵，不小心排除了有用变量。第二两个相对不相关变量，结合可能产生一个强相关变量。第二种问题应该利用主成分分析去获取有用的特征向量（变量的线性结合）
- 选择变量过程的一个问题：多重共线性：有两个或更多的独立变量是高度相关的。在回归模型，可能引发回归系数根据模型或数据的细微变化而不规律的变化
- 隐藏层越多，过拟合风险越大。
- 神经网络可能用于回归或分类。在回归模型中，简单的输出值被映射到一组真实数字，只需要一个输出神经元。分类系统输出一个神经元。如果类别未知，就要使用无监督NN比如：自组织映射
- 最高的办法遵守奥卡姆剃刀原理。对于两个性能相当的模型，自由参数更少的模型，泛化效果越佳。另一方面，不能牺牲效果来选择过度简化的模型。
- NN的训练算法有很多：必须停止的情况：误差率降低到了可接受水平、验证集的误差率开始变差或资源耗尽。最常见的NN学习算法：反向传播算法（backpropagation），这种算法使用了前文的梯度下降。
- 反向传播包括两个步骤：
  - 前向传播—将训练数据集通过网络，记录下神经网络的输出并计算出网络的误差
  - 反向传播— 将误差信号反向通过网络，使用梯度下降优化神经网络的权重
  - 存在问题：一次调整所有权重会导致权重空间的NN出现明显变化、随机梯度下降算法慢，对局部最小值敏感。对特定NN局部最小值。为了得到所需的全局最优化算法，两种流行的全局优化算法是粒子群优化算法（PSO）和遗传算法（GA）。
- NN可用的三种学习策略
  - 监督学习、	
    - 需要至少两个数据集：训练集由输入数据和预期输出数据组成。测试集只包含输入数据。这两个数据集的数据必须有标记，即数据模式是已知的
  - 无监督、
    - 一般用在没有标记的数据中发现隐藏结构：隐马尔可夫链，类似聚类算法
    - 流行框架：自组织映射（Self Organizing Map）本质上多维量度技术：另一个应用对股票交易的时间序列图标上色
  - 增强学习／强化学习
    - 策略由三部分组成
      - 一个指定NN如何决策的规则：例如技术分析／基本面分析
      - 一个区分好坏的奖赏功能：如赚／亏
      - 一个执行长期目标的价值函数
    - 在金融／游戏环境，强化学习策略特别有用，应为NN可以学习对特定量化指标进行优化。例如对风险调整收益的合适度量


##### [深度学习中的最小风险训练](http://chuansong.me/n/652376051971)

- 有监督学习指：训练样本不仅包含输入，同时包含对应标准答案输出。有监督学习的标准训练准则是极大似然估计，基本思想是一个好的模型应该尽可能使观测到的训练样本概率最大。
- 最小风险训练：使用损失函数来描述模型预测与标准答案之间的差异程度（损失）。并试图 寻找一组参数使得模型在训练集上损失的期望值（风险）最小。
- 在深度学习中使用最小风险训练有以下两大优点：
  - 适用于任意评价指标：任意定义在单个样本级的评价指标都可以作为最小风险训练中的损失函数。让训练出来的模型尽可能贴近用户需求
  - 适用于任意NN：最小风险训练不对模型框架做任何假设，无论是卷积NN还是CNN等

##### [解读GAN及进展](http://chuansong.me/n/1500459351517)

- GAN:Generative Adversarial Nets 生成式对抗网络：一方面将产生式模型拉回到了一直由判别式模型主场。也将对抗训练从常规游戏技能引到更一般领域。
- GAN基本框架：当判别器（Discriminator）不能区分真实数据x和生成数据G(z) 时，就认为生成器G达到了最优

##### [机器学习中贝叶斯基本理论、模型和算法](http://chuansong.me/n/1649815651232)

- 基本公式：我们用θ描述模型的参数，这个模型可以是神经网络、线性模型、或者SVM，参数都用θ来描述；大D是我们的训练集；π(θ)是先验分布，是我们看到数据之前对模型本身分布的描述；p(D|θ)是似然函数，给定一个模型θ的情况下描述这个数据的似然；我们的目标是想获得这个后验分布，是看到数据之后我们再重新看模型本身的分布情况。
- 机器学习中贝叶斯法则可以做什么
  - 预测问题：大M来描述model class，比如线性模型、非线性模型，model class里面有多个具体的模型，我们还是用参数θ表示
  - 还可以做不同模型的比较、模型的选择。比如说我们要做分类问题，到底是要选线性的模型还是深度学习的非线性模型，这是在做模型选择的问题。这个问题可以描述成这样：我们用M1表示一个model class，可能是一个线性模型，我们用M2表示另外一个model class，是非线性模型，我们在同样数据集D的情况下，我们可以去比较这两个量哪个大，这个量是描述的在M1下我们观察到训练集的一个似然，另外一个是在M2的情况下观察的数据集似然，可以通过这个比较看我应该选择哪一种模型，这是一个用贝叶斯方法做模型选择的一个基本的规则。
- 贝叶斯推理时，我们的输入是一个先验分布和一个似然函数，输出是一个后验分布
- 损失函数优化问题。比如说，我要做分类，我要训练神经网络，第一项是一个损失函数，它度量了在训练集的错误率；第二项是我们想加的正则化项，目的是保护这个模型避免过拟合、避免发散，这是一个基本框架。这些东西在机器学习里基本上和贝叶斯是不搭边的
- 增强学习/在线学习，它的目标是优化Regret/reward，也有一个目标函数来度量。
- 传统的SVM是找一个决策面，按照一定的最优准则来找。贝叶斯的思想是：我们可以有无穷多个决策面，但是每个决策面有一定的概率。

##### [基于社会媒体的预测技术(上)](http://chuansong.me/n/1703436)

- 社会媒体为预测技术提供了新的数据源
- 基于社会媒体的预测技术
  - 两方面作用：社会信号的采集／大众预测的融合
- 基于消费意图的挖掘的预测
  - 基于社会媒体的消费意图挖掘
    - 对于消费意图挖掘任务的领域相关等问题，提出了：**基于领域自适应卷积神经网络的社会媒体用户消费意图挖掘方法**。卷积神经网络对于解决该任务有以下两方面的优势：
    - 卷积神经网络中的卷积层可以以滑动窗口的方式捕捉词汇级语义特征，而马克斯池(max pooling)层则可以很好地将词汇级特征整合成句子级语义特征；
    - 卷积神经网络可以学习不同层次的特征表示，而一些特征表示则可以在不同领域间迁移。
  - 基于消费意图挖掘的电影票房预测
    - 推荐系统、产品销售预测等
    - 电影票房预测主流模型：线性预测模型／非线性预测模型：都有一个前提：收入与预测影响因素之间存在线性或非线性关系。首周预测：线性回归比非线性好，总票房，非线性高于线性。说明：上映前一周的数据与首周线性关系比较明显，此时线性预测好。随着时间推移，新因素加入以及偶然情况发生，使得线性关系不明显。线性回归模型的预测能力不如非线性。
- 基于事件抽取的预测
  - 不同于消费是从人的主观角度。事件从客观事实角度出发
  - 金融市场的预测研究分成：时间序列交易数据驱动和文本驱动两个不同方向。
    - 时间序列交易数据：最早用于建立预测模型的数据：包括股票历史价格数据等
    - 文本驱动：挖掘新闻报道中客观事实及大众情感等。
- 基于因果分析的预测
  - 与相关性相比，因果的确定性更强，如稀有事件等

##### [评论对象抽取综述](http://chuansong.me/n/2032451)

- 主流的提取技术：名词短语的频繁项挖掘、评价词的映射、监督学习方法及主题模型放方法。
- 评价对象抽取属于信息抽取的范畴：将非结构文本转换为结构化数据的技术。主要用于网络文本的意见挖掘。
- 如果挖掘在文本中已经出现的评论对象：主流方法4种：
  - 名词挖掘：从频繁的名词开始
    - 评论对象大都是名词或名词短语:词性标记得到语料中的名词
  - 评价词与对象的关系
- 监督学习方法
  - 信息抽取的研究提出了很多监督学习算法。其中主流的方法根植于序列学习（Sequential Learning，或者Sequential Labeling）
  - 目前最好的序列学习算法是隐马尔可夫模型（Hidden Markov Model，HMM）和条件随机场（Conditional Random Field，CRF）
- 主题模型（Topic Model）
  - 统计主题模型逐渐成为海量文档主题发现的主流方法。主题建模是一种非监督学习方法，它假设每个文档都由若干个主题构成，每个主题都是在词上的概率分布，最后输出词簇的集合，每个词簇代表一个主题，是文档集合中词的概率分布。
  - 目前主流的主题模型有两种：概率潜在语义模型（Probabilistic Latent Semantic Analysis，PLSA）和潜在狄利克雷分配（Latent Dirichlet Allocation，LDA）
  - 主题模型是基于贝叶斯网络的图模型


##### [用户画像User Profile之用户能力标签](http://chuansong.me/n/1980238)

- 微博作为最大的中文社交媒体，拥有数以“PB”（1024 TB）计的用户信息，从海量的用户信息中发掘每个用户的社交特性、潜在能力及兴趣等信息，是微博为用户提供更加人性化服务的基础。
- 用户画像体系。该体系涵盖**能力标签、兴趣标签、关系及亲密度、信用质量和自然属性**五大部分
- 每一个用户都是网络中的一个具备发布、传播、消费信息功能的节点。其中一部分节点具备发布优质原创信息的功能，并通过社交网络将信息快速传播，即**能力节点**；而其他大部分节点则偏重于消费信息，同时传播其感兴趣的信息，即**消费节点**。
- **用户标签体系、能力标签的应用场景、能力标签挖掘框架、关键技术点**四个方面对用户能力标签的整体挖掘框架和挖掘算法
- **用户标签体系**
  - 某个话题下的相关信息中聚合出一个或者多个具有代表性的词语作为标签，能够方便对用户与内容的查找与分析。
  - **在当前的三层用户标签体系中，共存在50多个一级标签，1000多个二级标签和近30万的三级标签**
- **能力标签的应用场景**
  - 其中两个典型的业务场景是“微博找人”和“热门微博”
  - 找人业务场景中，用户可以直接发现各垂直领域的专家账号，通过关注专家账号可以直接获取各垂直领域的优质内容。
  - 在热门微博业务场景中，内容流都出自于垂直领域的专家账号：一个账号通过发布某个领域的优质内容形成初步影响力，大数据计算出其所属领域后，热门微博会在对应领域进行内容推荐，使该账户逐步成长为专家账号，从而形成一个产品闭环。
  - 能力标签的主要作用是构建各种优质语料的重要基础数据源，通过能力标签圈定专家用户群体，提取出优质语料等相关信息；在大部分情况下，能力标签不直接在业务场景中展示
- **能力标签挖掘框架**
  - 首先通过**用户关系数据**(主要是分组，用于体现粉丝对于用户能力的认可度)、**用户内容数据**(主要是原创博文，用于体现用户自身的专业能力)、**用户行为数据**(主要是转、评、赞等互动信息，用于体现该用户在相关领域内的影响力)挖掘出用户的能力标签及其基础权重；其次通过引入**用户的自填信息、认证信息**作为能力标签权重的调权因子参与计算；接下来通过多个维度的定向挖掘系统和运营反馈系统进行能力标签的校正和增加能力标签的覆盖。最后，**将挖掘出来的用户能力标签及权重输出至用户能力标签库，供上层业务调用**。
- **信用质量和自然属性**
  - 标签词汇聚、用户影响力、时间窗口和时间衰减三个关键技术点
  - 标签词汇聚
    - 用户为关注对象打上的标签作为用户关系数据引入到挖掘过程中，由于标签属于UGC，就会造成同一个标签主题有多种不同的表达方式，将多种不同的表达方式聚合起来，形成一个标签集，并且映射到我们的标签体系中，可以有效地提升能力标签的准确率和覆盖率
    - 首先将分组信息通过分类模型划分为**强关系型**(同学、同事等)和**兴趣型**(互联网、财经等)两类，并将兴趣型分组信息作为我们的基础预料
    - 接下来通过**聚类、关联**等相关算法进行标签词(分组信息)的聚合；
    - 最后将聚合的标签集根据**相关程度等因子划分为高相关和低相关**两类
  - 用户影响力
    - **用户在某个特定标签下的影响力**，因此影响力计算的边界(如图5所示)是标签对应的兴趣用户群体（包含该标签的能力用户）
    - 具体地，我们将其它用户对某个用户原创博文的转、评、赞等互动行为作为基础数据，利用**pagerank迭代算法**进行该用户影响力的计算
    - 其中，同领域用户的影响力大小是由其它用户对相关博文的转、评、赞等互动行为按照一定的权重比计算得到的。
  - 时间窗口和时间衰减
    - 考虑到原创博文的消费价值和计算代价，对于用户内容数据，我们选取了用户近一段时期内的原创博文作为基础语料进行计算。
    - 关于时间衰减，我们结合牛顿冷却定律和微博的业务需求推导出相应的衰减公式，并通过衰减效果的对比，确定了相关衰减参数的数值，最终得出了用户能力标签内容权重的时间衰减函数
- 主要从**社交关系、原创内容、影响力**三个维度来识别用户的能力标签以及计算相应的权重，同时通过用户的自填信息、认证信息等其他信息进行调权。

##### [社会媒体挖掘](http://chuansong.me/n/1941613)

- 基于Web 2.0的思想和技术的互联网应用，支持用户创造和交换内容。
- 用户在社会媒体上分享、交流、联系、互动产生的海量数据，比如每天都查看女神的微信并点赞的数据，比如每天都发几张吃吃吃的美食图的数据
- 《微信社会影响力报告》《双十一剁手数据报告》《手游人群消费报告》
- 首先呢，要懂图、网络度量、网络模型和数据挖掘
- 度量和模型指导我们做出什么样的图，用什么标准解释图的含义
- 数据挖掘基本要素是数据获取、数据预处理、数据挖掘算法
- **社区分析**
  - 分析社区是如何形成、演变的，如何知道这个社区的质量？这就需要社区分析知识，比如社区发现算法
- **信息如何扩散**
  - 需要先搞懂信息的传播方式：羊群效应、信息级联、创新扩散和流行病
- **实际应用**
  - 分析社会网络中个人的影响力，典型的问题：微博中哪个大V最有号召力？
  - 在线为用户推荐个人和好友，典型的问题：微博是如何为你推荐好友的？
  - 分析用户个人行为，典型的问题：你今天会不会玩三国杀，玩完还会干什么？

##### **[语言分析技术在社会计算中的应用](http://chuansong.me/n/2294022)**

- **面向社会媒体的自然语言使用分析**
  - 传统的自然语言处理主要面向正式文本，例如新闻、论文等。这些文本遣词造句比较规范，行文符合逻辑，因此比较容易处理。
  - 自然语言处理技术按照处理目标分为几个层次：
    - 1）词汇层。主要是在词汇级别的处理任务，如中文分词、词性标注、命名实体识别等。（2）句法层。主要是在句法级别的处理任务，如针对句子的句法分析、依存分析等。（3）语义层。主要是在语义空间的处理任务，例如语义分析、语义消歧、复述等。（4）篇章层。主要是在篇章级别的处理任务，如指代消解、共指消解等。（5）应用层。主要是指利用自然语言处理分析技术完成的应用任务，如文本分类、信息抽取、问答系统、文档摘要、机器翻译，等等。
- **面向社会媒体的自然语言分析应用**
  - 社会预测
    - 产品销量、体育比赛结果、股市走势、政治选举结果、自然灾害传播趋势
    - 社会媒体中关于候选人的提及率就是很好的预测指标，例如根据Facebook上的支持率就能够成功预测2008年美国总统大选结果

##### **[个性化推荐](http://chuansong.me/n/2823936)**

- 推荐系统的研究和生产实际，基于以上这个关于推荐的肤浅定义，把整个系统的模型简化成了——预测用户对于某个事物的喜爱，也就是人们常说的Rating Prediction的问题[3]。Collaborative Filtering，特别是基于Matrix Factorization[2]和Latent Factor Model[1]的各种方法
- 协同过滤，辅佐以Machine Learning的很多手段（比如Tree-based Models)，常常能够在Rating Prediction这个问题上有不俗的成绩
- 原因很直观，要想优化用户喜好，那就必然强调用户的历史行为。协同过滤或者是机器学习导向的算法，都试图充分挖掘单个用户以及群体用户的喜好，并且加以推崇到极致（Optimization)。

##### **[为学者写的提高生产力的方法技巧](http://chuansong.me/n/1703442)**

- 优化交易成本
  - 保证你生活道路上的阻力最小。因为最小的阻力意味着最大的生产力。

##### **[英文论文写作](http://chuansong.me/n/2833039)**

- 段落就是为读者描述、解释了一个核心思想的一段具有逻辑性的文字
  - A logical “unit” of text which develops/explains an idea to the reader.
- 文章就是由多个具有逻辑性的段落前后关联组成的。
  - An article is a chain of logical units/paragraphs.

##### **[顺滑：让语音识别更流畅](http://chuansong.me/n/1443061951148)**

- 自动语音识别（ASR）得到的文本中，往往含有大量的不流畅现象
- 不流畅现象主要分为两部分，一部分是ASR系统本身识别错误造成的，另一部分是speaker话中自带的
- NLP领域主要关注的是speaker话中自带的不流畅现象，ASR识别错误则属于语音识别研究的范畴。顺滑(disfluency detection)任务的目的就是要识别出speaker话中自带的不流畅现象。
- 对于顺滑任务，研究分4类：序列标注方法、句法和顺滑联合方法、基于RNN的方法、基于seq-to-seq的方法。目前性能最好的是基于RNN的方法和基于seq-to-seq的方法
  - 序列标注方法
    - 这类方法可以分为两大类。一类是基于词的方法。这类方法的做法是利用序列标注模型，给句子中每个词赋一个标签，最后根据标签来判断词的类型。
    - 另一类是基于块（chunk）的方法，这类方法的输出不再是与输入等长的标签序列，而是一个个带标签的短语块，这类方法的一个优点是可以利用块级别的特征
    - 传统的序列标注模型通过设计复杂的离散特征，可以在一定程度上解决长距离依赖问题，但是受限于训练语料的规模，往往会面临稀疏性的问题
  - 句法和顺滑联合方法
    - 联合方法需要同时进行句法和顺滑分析，相对于序列标注等方法，其速度会比较慢。
  - 基于RNN的方法
    - Recurrent Neural Network（RNN）是为了对序列数据进行建模而产生的，其在理论上能很好的解决长距离依赖问题，因此一些研究者尝试将RNN网络应用到顺滑任务中。LSTM(Long Short Term Memory networks)是一种特殊的RNN网络，它可以有效减轻简单RNN容易出现的梯度爆炸和梯度消散问题
    - 直接用双向LSTM的隐层输出来对每个位置的词进行分类，判断其是否为不流畅词，其性能已经超越了之前最好的序列标注方法以及句法和顺滑联合方法。
    - 这种直接分类的方法没有考虑输出标签之间的联系，当输出标签之间存在强依赖性时，这种分类方案可能会导致 标签偏置(label bias)的问题。
    - 为了解决标签偏置的问题，我们尝试采用LSTM-CRF模型。在LSTM-CRF模型中，首先通过双向LSTM学习到每个位置的特征表示，然后将学习到的特征表示直接送到一个线性CRF模型。从实验结果来看，LSTM-CRF模型的F1值要比LSTM高一个点左右
  - 基于seq-to-seq的方法
    - 采用seq-to-seq方法主要有两个动机。一个是seq-to-seq框架在编码阶段会对输入句子学习一个全局的表示，该全局表示有助于解决长距离依赖问题
    - 另一方面，seq-to-seq方法本身可以被看做一个基于条件的语言模型，原始的输入句子相当于语言模型的条件，解码阶段相当于一个语言模型的生成过程，这样就有一定得能力保证生成句子的句法完整性
    - 传统的encode-decode框架显然不能满足顺滑任务的要求，总结起来，其主要有三个局限
      - 一是其每步生成新词的时候，都会在一个固定的词表中去选择一个概率最大的词，这样就可能会生成一个不在原始句子中出现的词，
      - 二是其只能在固定的词表中去选词，如果原始句子中出现了一个不在词表中的词，那么这个词就肯定不会被生成，这明显是不符合顺滑任务要求的
      - 最后一个原因是其无法保证生成词的有序性。

##### **[Dropout分析](http://chuansong.me/n/1519459851816)**

- 过拟合(Overfitting)是深度神经网络（DNN）中的一个重要的问题：该模型学习仅对训练集合进行分类，使其自身适应于训练示例，而不是学习能够对通用实例进行分类的决策边界。
- 已经提出了许多过拟合问题的解决方案，其中，Dropout因为其简明且以经验为基础的良好结果而占据主流
- Dropout的思想是训练DNNs的整体然后平均整体的结果，而不是训练单个DNN。DNNs以概率p丢弃神经元，因此保持其它神经元概率为q=1-p。当一个神经元被丢弃时，无论其输入及相关的学习参数是多少，其输出都会被置为0。丢弃的神经元在训练阶段的前向传播和后向传播阶段**都不起作用**：因为这个原因，每当一个单一的神经元被丢弃时，训练阶段就好像是在一个新的神经网络上完成。
- Dropout在实践中表现良好，是因为它在训练阶段阻止了神经元的共适应。
- Dropout如何工作：
  - Dropout以概率p关闭神经元，相应的，以大小为q=1-p的概率开启其他神经元。**每个单个神经元有同等概率被关闭。**

##### **[基于协同过滤的中文零指代消解方法](http://chuansong.me/n/1389485951863)**

- 指代消解是信息抽取不可或缺的组成部分。在信息抽取中，由于用户关心的事件和实体间语义关系往往散布于文本的不同位置，其中涉及到的实体通常可以有多种不同的表达方式
- 中文的零指代是指代现象中的一种，是指代现象中的一种特殊情况。它是指在篇章中，读者能够根据上下文的关系推断出来的部分经常被省略，被省略的部分在句子中又承担相应的句法成分，并且回指前文中的某个语言单位
- 被省略掉的部分称为零指代项或者零代词
- 中文零指代消解系统中，都统一采用了单候选模型（single candidate model）
- 基于物品的协同过滤算法，item-based collaborative filtering algorithm 
  - 简单加权法 (simple weighted average)”来进行推荐过程
  - 假设我们有若干物品![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz_png/58FUuNaBUjqFbHYxFRyddWG6zubFzxtkowvGicnqtarOlWSiawb7Ec6tLnCpABh4THz7qduQBPMwbVMTYC1BoswA/0?wx_fmt=png)，并知道这些物品彼此之间的关联程度（相似度）![Rendered by QuickLaTeX.com](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz_png/58FUuNaBUjqFbHYxFRyddWG6zubFzxtkbUk6Al1ic3umnx4lGjC1MyuKyBdcVh8lickN9RWtVHSiaydF7zjepE8lQ/0?wx_fmt=png)，那么一个用户对一件物品的打分就可以通过其他相似物品通过加权平均的算法来近似获得
  - 也就是说，这个方法能够利用物品之间彼此的相似度，关联度来更好地衡量用户对一件物品的倾向性

##### **[两种阅读理解模型框架的概要介绍--有数据集／代码](http://chuansong.me/n/1344812751660)**

- 为了考察⼈类的阅读理解能⼒，往往采取提问的⽅式，即：给定⼀篇⽂本和与⽂本相关的问 题，要求给出该问题的答案
- 神经⽹络模型成为阅读理解任务中的主流模型。下⾯简要的介绍其 中两种具有代表性的模型
  - Attention Reader[2] 在处理的时候，⾸先采⽤双向 RNN 分别表⽰⽂本（看做⼀个“长句⼦”）和问 句，再利⽤ attention 机制寻找⽂本表⽰中同问句相关的信息，最后根据相关程度提取⽂本信息做 分类并给出预测的答案。
    - **表示**：使⽤双向 RNN（LSTM Cell）获取⽂本表⽰和问句表⽰。其中问句表⽰分别由双向 RNN 两个⽅向各⾃最后⼀个时刻的隐层状态（图中左上⾓双向 RNN 中的橙⾊向量）拼接⽽ 来
    - **Attention**: 使⽤ Attention 机制获得⽂本中各个时刻的隐藏状态向量同问句表⽰之间的相关 程度（也就是权重，最简单的做法就是向量点乘），在图中⽤红⾊线条表⽰（越长表⽰相关程 度越⾼）
    - **输出层**：⽂本各个时刻隐藏状态向量乘以对应时刻的权值，加和，获取“提取后的⽂本信息”， 过 softmax 在词表上进⾏分类；
    - AttentionReader中，核⼼的思路是通过动态的attention机制从⽂本中“寻找相关信息”，再做 依据该信息给出预测结果。
  - Attention-Sum Reader，该模型直接利 ⽤ attention 机制基于问句表⽰在⽂章中寻找最相关的词作为答案。
    - Attention-Sum Reader 直接利⽤ attention 机制获取⽂本中各个位 置作为答案的概率（需要将同⼀词语在⽂本中不同位置的概率加和作为该词语成为答案概率），⽽ 不再同 Attention Reader 最后⼀步那样，通过分类（softmax）获取各个词语作为答案的概率。

##### **[基于事件的金融市场预测研究](http://chuansong.me/n/1281017951054)**

- 金融市场的预测研究可以追溯到在研究不确定性问题时提出的选美理论，即在金融市场投资问题上，不要去买自己认为能够赚钱的金融品种，而是要买大家普遍认为能够赚钱的品种，哪怕那个品种根本不值钱，也就是说投机行为就是建立在对大众心理的猜测上
- 基于事件的预测则是从较为客观的事实角度出发进行预测，新闻媒体中报道的一些事件会对人们的决策产生影响，而人们的决策又会影响到他们的交易行为，这种交易行为最终会导致金融市场的波动。
- 重要事件都会导致股票市场的剧烈震动，如果能够及时准确地获取这些重要事件势必会对金融市场波动的预测起到重要帮助作用。
- 有效市场假说（ Efficient Market Hypothesis）
  - 金融产品的价格可以充分反映出关于该资产可获得的所有信息，即“信息有效”，而每个人都一定程度上可以获得这些相关信息
  - 这一假说可以作为基于事件抽取的股市预测的理论基础，因为企业发生的事件是与其最相关的信息，而且也是大众普遍可以获取到的信息。
- 金融市场的预测研究可以分成两个不同的研究方向，
  - 一是时间序列交易数据驱动的金融市场预测，
    - 最早用于建立预测模型的一类数据，主要包括股票历史价格数据、历史交易量数据、历史涨跌数据等。传统的金融市场预测研究中，金融领域学者多从计量经济学的角度出发进行时间序列分析，进而预测市场的波动情况
  - 二是文本驱动的金融市场预测。
    - 主要是挖掘新闻报道和社会媒体中报道的客观事实以及大众的情感波动。前人的很多研究工作表明金融领域新闻会一定程度上影响股票价格的波动
    - 尝试抽取文本中的命名实体和名词短语来扩展词袋模型。
- 基于情感分析的金融市场预测主要是从主观情感角度出发进行预测，而基于事件的金融市场预测则是从客观事实角度出发进行预测，二者可以相结合，优势互补，取得更加精准的预测结果。
- 基于事件的股票预测
  - 以往工作存在一个共性的问题，没有捕捉到文本中的结构化信息，而这一信息对于股票涨跌预测非常重要
  - 事件表示学习
    - 由于传统的 One-hot 高维特征表示方式会使得事件特征异常稀疏，从而不利于后续的研究和应用，因此，我们提出了三种全新的事件表示方式
      - 第一种离散模型是基于语义词典对事件元素，进行泛化，进而缓解事件的稀疏性。第二种连续向量空间模型则为每一个事件学习一个低维、稠密、实数值的向量进行表示，从而使得相似的事件具有相似的向量表示，在向量空间中相邻。第三种模型在连续向量空间模型的基础上引入外部知识，进一步增强了事件向量的表示能力。
    - 基于离散模型的事件表示学习方法
      - 历史上发生的事件大多数都很难以再次发生，因此会导致事件具有严重的稀疏性，离散模型的目标是对同一事件的不同表达进行归一和泛化
      - 泛化过程包含两个步骤。首先，本文从 WordNet 中找到事件的施事者和受事者中名词的上位词将其泛化。例如，本文利用“微软”的上位词是“IT 公司”将其替换掉。随后，本文找到事件元素中的动词，并用 VerbNet 中该动词所属类别的名词替换掉改动词，从而对其进行泛化。例如，“增加”在 VerbNet 中所属的动词类别名称为 multiply
      - 下面给出一个事件泛化的完整例子，给定句子“Instant view: Private sector adds 114,000 jobs in July.”，可以抽取出事件（ Private sector, adds, 114,000 jobs）将其泛化后的结果是（ sector, multiply class, 114,000 job）。类似方法也曾被 Radinsky[126] 提出用来做因果事件预测任务上。
    - 基于连续向量空间模型的事件表示学习方法
      - 离散模型方法简单且有效，但是也存在着两个重要的局限性：其一，WordNet， VerbNet 等语义词典词覆盖有限，很多词难以在语义词典中找到相应记录。其二，对于词语的泛化具体到哪一级不明确，对于不同应用可能会有不同要求，很难统一
      - 设计了一个全新的张量神经网络来学习事件的结构化向量表示，事件的每一个元素及其所扮演的角色都会被显式地建模学习
    - 融入背景知识的事件表示学习方法
      - 事件的向量表示能够学习到事件中包含的语义信息，缓解离散事件的稀疏性，但是也存在一定的局限性。一方面，对于句法或语义上相似的事件，如果它们不包含相似的词向量，那么事件的向量表示可能无法捕获它们时间的相似关系；另一方面，如果两个事件包含相似的词向量，例如“ Steve Jobs quits Apple”和“ John leaves Starbucks”，可能具有相似的向量表示即使它们毫无关联。其中一个重要原因是，在训练事件表示时缺少背景知识。如果我们知道“ Steve Jobs”是“ Apple”的 CEO，而“ John”可能是“ Starbucks”的一位顾客，那么模型就能学到完全不同的向量表示。
  - 预测模型
    - 我们将股市涨跌预测看成二元分类问题，即判断股价未来走势是上涨，还是下跌。具体而言，预测模型的输入是上一节中抽取到的具有结构化信息的事件，输出是预测当天股市的收盘价相对于开盘价是上涨还是下跌。
    - 基于深度神经网络模型的股市涨跌预测方法
      - 在股市预测任务上，大多数前人工作均采用线性分类器。直观上来讲，现实世界中发生的事件与股票涨跌之间的关系应该是复杂的隐含的非线性关系。因此我们提出了基于深度神经网络的预测模型
      - 神经网络的训练算法采用的是经典的反向传播方法
    - 基于卷积神经网络模型的股市涨跌预测方法
      - 值得注意的是对于历史事件而言，尽管其影响力有所衰减，但还是对股价的波动有一定的影响作用。
      - 前人工作很少会定量分析长期历史事件对股市波动的影响，尤其少见将长期事件和短期事件结合起来预测股市波动的工作。
      - 将长期历史事件看成是一个事件序列，利用卷积神经网络（ convolutional neural network， CNN）将输入的事件序列进行语义合成，然而利用网络中的 Pooling 层抽取出信息含量最丰富的事件作为特征，网络中的隐含层用来学习股市波动和事件之间的复杂关系。
      - 模型的输入是连续的事件向量序列，事件按照报道时间先后顺序排列。每一天的事件序列作为一个单独的输入单元(U)。模型的输出是一个二元分类，其中输出的类别+1代表预测当天股市的收盘价相对于开盘价是上涨，类别-1代表预测当天股市的收盘价相对于开盘价是下跌。

##### **[文本蕴含相关研究简介](http://chuansong.me/n/1238576451130)**

- 在获取了文本的语义后，获得了它们之间的推理关系，这些文本便不再互相孤立，而是彼此联系起来，构成一张语义推理网络，从而促使机器能够真正理解并应用文本的语义信息。
- 文本间的推理关系，又称为文本蕴含关系
- 文本蕴含定义为一对文本之间的有向推理关系，其中蕴含前件记作T(Text)，蕴含后件记作H(Hypothesis)。如果人们依据自己的常识认为H的语义能够由T的语义推理得出的话，那么称T蕴含H

##### **[基于词的神经网络中文分词方法](http://chuansong.me/n/1178530451667)**

- 中文分词是很多中文自然语言处理任务的第一步。中文分词的方法中，认识程度最高的是基于字的分类或序列标注方法。对于输入字序列，这一类方法解码出代表词边界的标签，然后从这些标签中恢复出分词结果。基于字的方法具有简单高效的特点，也有诸如无法直接利用词级别特征的缺点。不同于基于字的方法，基于词的中文分词方法能够在解码过程中获得部分的分词结果，因而能够充分利用词级别的特征。
- 深度学习的浪潮给自然语言处理研究带来诸多新思路。其中一项非常重要的思路是使用稠密向量与非线性的网络表示自然语言。在这样的背景下，基于词的神经网络中文分词方法成为一个很有趣的研究问题。如何表示中文分词中的词向量，词向量表示能否与解码算法很好的融合等都是基于词的神经网络中文分词方法要回答的问题
- SRNN：semi-CRF算法与神经网络进行结合
  - semi-CRF看起来比较陌生，但它的“近邻”——线性链条件随机场(linear chain CRF或CRF)相信大家都很熟悉
  - CRF基于马尔科夫过程建模，算法在随机过程的每步对输入序列的一个元素进行标注。而semi-CRF则是基于半-马尔科夫过程建模，算法在每步给序列中的连续元素标注成相同的标签。
  - 深度学习的一大优势在于强大的特征表示能力

##### **[中文语义依存图分析](http://chuansong.me/n/1132634551987)**

- 要让机器能够理解自然语言，需要对原始文本自底向上进行分词、词性标注、命名实体识别和句法分析，若想要机器更智能，像人一样理解和运用语言，还需要对句子进行更深一层的分析，即句子级语义分析
- 语义依存分析是通往语义深层理解的一条蹊径，它通过在句子结构中分析实词间的语义关系（这种关系是一种事实上或逻辑上的关系，且只有当词语进入到句子时才会存在）来回答句子中“Who did what to whom when and where”等问题。
- 树结构融合依存结构和语义关系
- 对于语义依存树表示体系存在的问题，我们采用的解决方案是用语义依存图分析代替语义依存树分析。形式上类似于依存语法，但必要时突破树形结构
- 语义依存树与语义依存图的主要区别在于，在依存树中，任何一个成分都**不能**依存于两个或两个以上的成分，而在依存图中则**允许**句中成分依存于两个或两个以上的成分。且在依存图中**允许**依存弧之间存在交叉，而依存树中**不允许**
- 目前依存分析领域两大主流方法分别是基于转移（Transition-based）和基于图（Graph-based）的依存分析。基于图的算法将依存分析转换为在有向完全图中求解最大生成树的问题，是基于动态规划的一种图搜索算法。该算法由McDonald等于2005年提出，是一种全局最优的搜索算法。基于转移的依存分析算法将句子的解码过程建模为一个转移序列的构造过程。其依存分析模型的目标是通过学习得到一个能够准确的预测下一步转移动作的分类器。
- **基于**Stack LSTM的分类器
  - 解决了转移系统的问题，接下来就需要选择一个合适的分类器在每个转移状态下预测出下一步要执行的转移动作
  - 传统LSTM对一个从左到右的序列进行建模，此前已经在此基础上发展出了双向LSTM和多维LSTM等模型。

##### **[基于表示学习的信息抽取方法浅析](http://chuansong.me/n/1105521251760)**

- 信息抽取(Information Extraction)是自然语言处理任务中的一个重要研究方向,其目的是从自然语言文本中抽取实体，关系和事件等事实信息，并形成结构化的数据输出
- 评测中，语言学专家往往都已将这些事实信息预先定义为不同的类别，人们只需要识别这些事实信息并将其分类即可，例如实体可分为：人名、地名和机构名。
- 将信息抽取任务转化为序列标注任务，通过不同的神经网络结构学习出词汇的表示和句子的表示，并在此基础上进行事实信息的识别和分类，与传统机器学习方法相比，这些表示信息不需要人工进行特征选择，也不需要依赖于现有的自然语言处理工具，因此不但节省人工也能避免pipeline系统中所产生的误差积累。
- 基于表示学习的信息抽取方法为主题，重点介绍文本实体抽取、关系抽取和事件抽取的任务描述和研究方法，并且每个任务将给出一篇相关论文做具体讲解.（本文中将会用到LSTM [7]，CNN [11]  和Softmax [6]   三种基本的神经网络结构
- 基于表示学习的命名实体抽取
  - 命名实体识别任务旨在识别出待处理文本中三大类（实体类、时间类和数字类）、七小类（人名、机构名、地名、时间、日期、货币和百分比）命名实体
  - Neural Architectures for Named Entity Recognition
    - 主要特点有二：第一点是设计并实现了一个 基于表示学习的CRF layer [9] , 该层有效捕获了标签间的依存信息，例如（e.g., I-PER 标签不能跟在 B-LOC 标签后）。二是用一个Character-based lstm去学习单词的字符级表示，该表示可以很好的解决语料中未登录词的问题
- 基于表示学习的实体关系抽取
  - 实体关系识别是一项自顶向下的信息抽取任务,需要预先定义好关系类型体系,然后根据两个实体的上、下文预测这两个实体之间的语义关系属于哪一种关系类别,其目的在于将实体间的抽象语义关系用确定的关系类型进行描述。我们一般只对同一句话中的两个实体进行关系识别,因此这个任务可以描述为:给定一个句子 s 以及 s 中的两个实体 Entity1 和 Entity2,预测 Entity1 和 Entity2 在句子 s 中的关系类型 rel,rel 的候选集合是预先定义好关系类型体系 R
  - Relation Classification via Convolutional Deep Neural Network
    - 主要思想是将实体对的表示分为两类不同特征表示，一类是词典特征，一类是句子全局特征。
    - 该系统输入是一个带有实体对标注的句子，句子中的每一个词都会经过一个look up层从之前pre－trai的word embedding中找到对应的向量表示，之后用这些向量表示来学习词典特征和句子特征，最终将这两种特征串联起来通过一个softmax layer进行分类。
- 基于表示学习的事件抽取
  - 事件抽取是信息抽取领域的一个重要研究方向。事件抽取主要把人们用自然语言表达的事件，以结构化的形式表现出来。根据定义，事件由事件触发词（Trigger）和描述事件结构的元素（Argument）构成。图1结合ACE的事件标注标准详细的表述了一个事件的构成。其中,“出生”是该事件的触发词，所触发的事件类别（Type）为Life，子类别（Subtype）为Be-Born。事件的三个组成元素“毛泽东”、“1893年”、“湖南湘潭”，分别对应着该类（Life/Be-Born）事件模板中的三个元素标签，即：Person、Time以及Place
- 信息抽取的相关概念，包括命名实体识别、关系识别和事件识别
  - 信息抽取包含多个子任务，这些相关任务之间往往存在着一定的约束和限制,命名实体识别的准确与否是影响关系抽取和事件元素识别的一个重要因素,如果可以对这些子任务的内在机理和特征进行融合,必然会使信息抽取技术的性能得到全面的提高。
  - 目前来看，基于表示学习的信息抽取技术的抽取策略都要依赖于一定的类别体系，这些类别往往都是由语言学专家预先设定，然而无论体系多么丰富，都会在新语料中遇到新的实体、关系或事件类型，超出之前的设定
  - 国内针对中文的关系抽取研究起步较晚，并且缺少相关评测支持

##### **[真实信息发现任务及方法介绍](http://chuansong.me/n/1070406151854)**

- 面对同一个对象(object)，多个信息源可能提供相互矛盾的信息或陈述(statement)。如何从多信息源的矛盾信息中识别真实信息，是一个有实用价值的研究问题
- 学术界将该问题称为真实信息发现，英文名称为Truth Discovery或者Truth Finding。该研究任务的输入是对物体或事件的描述信息，系统对信息进行可信度评估、从而筛选出真实信息
- 数据融合的应用需求，真实信息发现的研究多集中于结构化数据，重点关注信息源可靠性与信息可信度之间的相互作用及信息自身难易程度等
- 信息可信度的相关影响因素及模型算法
  - 本模型将信息源的可靠度进行向量化表示，并作为记忆进行长期存储。当信息的可信度发生变化时，记忆单元中的信息源的可靠度也随之修正。
  - 本模型中采用LSTM完成对信息的表示和真实值的预测。LSTM通过输入门i、遗忘门f及输出门o控制前序信息的隐含层对当前信息的抽象表示的影响。
  - LSTM模型的特点与本任务的需求相契合，我们采用基于记忆网络的LSTM模型完成对真实值的预测。
  - 记忆网络(MemoryNetwork)由记忆存储单元(M)及四个模块构成，包括输入模块(I)、生成模块(G)、输出模块(O)及反馈模块(R)。

##### [面向回复生成的Seq2Seq模型概述](http://chuansong.me/n/1011329851155)

- 越来越多深度学习模型被应用到各种自然语言处理任务中。Sequence to sequence是一种建模两个序列间关系的通用深度学习模型，这种高度抽象的特性，使得其可以应用到多种序列化的自然语言处理任务
- 模型结构
  - Sequence to sequence 模型的输入和输出分别称为输入序列和输出序列。一个标准的Sequence to sequence 模型通常由两部分组成：Encoder和Decoder（如图1）。Encoder部分负责依次读入输入序列的每个单位，将其编码成一个模型的中间表示（一般为一个向量），在这里我们将其称为上下文向量c。Decoder部分负责在给定上下文向量c的情况下预测出输出序列。
  - Encoder和Decoder部分通常选择用Recurrent Neural Network(RNN)来实现。
  - 这样做的原因基于以下几点：首先，RNN作为一种循环结构可以方便地处理可变长序列的数据。其次，由于RNN中的隐层状态随着按时序读取的输入单元而不断发生变化，因此它具有对序列顺序的建模的能力，体现在自然语言处理任务中，即为对词序的建模能力。而词序也恰恰是自然语言处理任务中需要建模的重点。最后，RNN可以作为一个语言模型来预测出给定前文的基础上下一个字符出现的概率，这个特性使得其可以应用在各种文本生成任务中预测出语法正确的输出序列，从而实现Decoder的功能。
- 模型改进
  - 基本Sequence to sequence 模型基础上，我们还可以结合一些其他结构来提升其在某些任务中的效果。在机器翻译任务上，Cho等人在Decoder部分进行了改进，为Decoder RNN的每个结点添加了由Encoder端得到的上下文向量作为输入，使得解码过程中的每个时刻都有能力获取到上下文信息，从而加强了输出序列和输入序列的相关性(
  - 加入了Attention机制，在解码输出序列的每个单位时动态计算出Encoder所有隐层状态对当前解码过程的重要性，然后按重要性大小对隐层状态赋权，进一步计算出Encoder隐层状态的加权和作为动态的上下文向量。这个过程类似于统计机器翻译中的对齐过程
- 作为一种建模序列间关系的通用深度学习模型，Sequence to sequence模型成功应用到了自然语言处理的多个领域。在机器翻译任务中，带有Attention机制的双向RNN模型取得了和传统短语级统计机器翻译模型可比的效果。在回复生成任务中，Sequence to sequence模型可以成功地生成语句通顺内容相关的回复，在一定程度的改进后更是成功地从单轮回复生成推广到了多轮

##### **[微博用户传播行为预测](http://chuansong.me/n/982487051530)**

- 大量的用户生成信息(user generated content，简称UGC)也带来了诸如信息过载、虚假信息泛滥等问题，微博信息传播预测的研究为解决这些问题提供了可能。
- 微博信息传播预测是指在掌握现有信息传播形态的基础上，依照一定的方法和规律对未来的信息传播趋势进行测算，以预先了解信息传播的最终过程和结果，为信息传播的干预提供依据。
- 例如：研究用户的在线行为以及传播行为规律将有助于网络公司准确地把握用户的偏好，并将可能感兴趣的话题信息、其他用户或者用户社群推荐给该用户[4]。
- 微博信息传播预测中的两个主要元素是信息和用户，信息与用户之间存在相互作用。目前，根据预测任务的侧重点不同，可概括为“以用户为中心”、“以信息为中心”和“以信息和用户为中心”三个方面[5]。本文主要关注以用户为中心的研究，即以用户的兴趣和行为建模为基础，主要任务是用户传播行为预测。
- 具体来说，预测用户是否会参与某信息的传播。在微博中，用户对信息的传播行为主要指转发行为[6,7]。本文将以微博中用户的转发行为为例，介绍微博用户传播行为分析与预测的研究工作
- 转发行为影响因素
  - 首先，对微博用户转发行为预测问题进行形式化定义
  - 核心用户 *u* 是否会转发用户 *p* 在 *t* 时刻发布的微博信息 *v* ，*ruvp*可以由多种形式来表示，比如布尔值、相对顺序、概率值等
  - 用户的转发行为是多种因素共同作用的结果[
  - 通过对收集到的转发信息进行归类总结，列出用户转发行为产生的原因。本文综合以往研究内容，将用户转发行为产生的影响因素概括为两类——信息内容因素和群体影响因素
    - 信息内容因素主要包括信息内容自身特点以及信息内容与用户兴趣的吻合程度：
      - 前者包括信息内容的流行程度(是否热门话题)、信息内容的丰富性(是否含有多媒体、图片等)
      - 后者指用户是否对此类信息感兴趣
    - 群体影响因素主要包括信息发布者对用户的影响以及其他信息转发者对用户的影响。
- 转发行为预测方法
  - 用户转发行为预测是指通过一定的手段学习用户的兴趣和行为规律，从而对未知的用户转发行为进行预测。按照预测基本假设的不同，用户转发行为预测方法可分为基于用户过往行为的预测、基于用户文本兴趣的预测、基于用户所受群体影响的预测以及基于混合特征学习的预测。主要使用的模型包括：协同过滤模型、主题模型、因子图模型以及分类模型等。
  - 基于用户过往行为
    - 基于用户过往行为的预测方法依据用户在预测时间点前的过往行为，预测用户未来的行为
    - 该方法认为：用户的兴趣短时间内不会改变，用户转发微博的行为受用户兴趣所驱动[10]。因此，可充分利用已知的用户偏好或行为，预测未知的用户偏好或行为。用户过去转发了某些微博信息，则很可能还会对类似内容的微博信息感兴趣
    - 协同过滤模型预测用户的转发行为。协同过滤的概念来源于推荐系统，目前应用最广泛的是矩阵分解技术，其核心思想是：假设用户的兴趣只受少数几个因素的影响，因此将稀疏且高维的“用户-物品”矩阵分解为两个低维矩阵，通过用户对物品的评分信息来学习用户特征矩阵*ui*∈*RK *和物品特征矩阵 vj∈*RK*，最后重构低维矩阵预测用户对物品的评分![img](http://read.html5.qq.com/image?src=forum&q=5&r=0&imgflag=7&imageUrl=http://mmbiz.qpic.cn/mmbiz_png/58FUuNaBUjq7GlKicsD47mDZpibB15dDfKrtTy59LQjATDTaWp4FxzATafiaQmrE3DPFRfGwHANNyK1ib9gSj9ia59A/0?wx_fmt=png)。
    - 类似的，用户与微博信息可构建“用户-信息”矩阵，此矩阵中，元素的值为1表示用户转发该微博信息。然而，不同于传统的商品推荐，用户转发信息的数据集中，由于新的信息不断出现，“用户-信息”矩阵是非常稀疏的，存在较严重的冷启动问题。因此，后续基于用户过往行为的预测研究工作致力于在传统协同过滤模型基础上融入丰富的特征，如用户属性特征、微博信息特征以及传播结构特征等
    - 关键词和主题抽取，把“用户-信息”矩阵转化为“用户-关键词”或“用户-主题”矩阵，在一定程度上缓解了数据稀疏导致的新信息冷启动问题
  - 基于用户文本兴趣
    - 基于用户文本兴趣的预测方法认为，用户对某信息的转发行为源于用户对微博文本内容的兴趣
    - 此类方法将用户历史微博信息视为该用户的伪文档，通过对用户进行文本兴趣建模，预测用户对未知信息的转发行为
    - 许多研究通过词袋模型 (bag-of-words) 对用户文本以及微博信息进行向量表示，然后计算文本相似度，相似度越大，用户转发该信息的可能性越大[7]。另一方面，以隐含狄利克雷分布 (latent Dirichlet allocation，简称LDA) [13] 为代表的主题模型及其变型被广泛应用于社会媒体用户文本兴趣建模任务中
    - 用户的文本兴趣是随时间变化的，因此提出了基于分层狄利克雷过程(HDP)的非参数贝叶斯模型。该模型不仅能够对用户兴趣进行动态的主题建模，还融合了其他影响因素，如用户所受其他用户的影响
  - 基于用户所受群体影响
    - 基于用户所受群体影响的预测方法的基本假设是，用户转发行为的产生主要由于其所受群体的影响。
    - 一般而言，用户转发行为所受的群体影响分为两方面：一方面来自于信息发布者的影响，另一方面来自于群体中其他人的转发行为的影响[
    - 对照组和干扰组，验证了用户之间局部影响力的存在。基于局部影响力和全局影响力的逻辑回归模型，有助于提升用户转发行为预测的结果。
    - 实验还表明：用户转发某微博的可能性与其好友中转发该微博的人数成正比，而与这些好友形成的社交圈数成反比。
    - 因子图 (factor graph) 是对函数因子分解的表示图，在社会网络建模中得到了广泛应用。它将多变量函数描述为“二分图”，每一个因子图都包含两类节点——变量节点 (variable node) 和因子节点 (factor node)，边只连接不同类型的节点
    - 通常，通过和积算法(sum- product algorithm)求解各个变量的边缘分布。假设*rp，u，v *∈ {-1，1} 表示用户 *u *是否转发由 *p* 发布的微博 *v，R *= {*rp，u，v *} 为待预测的用户转发行为集合，*G *为当前网络拓扑结构，则预测问题可转化为：在给定 *G *和 *A *的情况下，求解所有待预测转发行为的最大联合条件概率问题
  - 基于混合特征学习
    - 基于混合特征学习的预测方法将转发行为预测视为二元分类问题，分析影响用户转发行为的因素并作为特征，然后选择适当的分类器训练分类模型
    - 常见的特征可概括为独立特征和关系特征。独立特征指核心用户、微博以及微博发布者各自的特征；关系特征指三者之间的相互作用特征，如用户与微博发布者之间的社会关系、用户对微博内容的感兴趣程度以及微博发布者在该信息主题的权威度
    - 如果微博发布者与核心用户有较亲密的社会关系，那么核心用户对于微博发布者的信息更容易产生转发行为，社会关系特征可体现于两者是否是双向好友、两人历史上微博互转的频度等
    - 此类方法的关键在于各种特征的选择和组合。在对比各种因素对用户转发行为的影响时，常用的方法是基于特征递增法(“add-one-feature-in”)或特征排除法(
    - 用户转发行为的影响因素划分为基于社会关系的特性、基于内容的特征、基于发布者的特征，训练多种分类器(决策树、SVM、逻辑回归)，并使用特征排除法对比了各类特征的有效性，并说明社会关系特征相对于其他特征更加重要
    - 二元分类的排序函数对某微博的可能转发者进行top-K排序，发现：如果用户与微博发布者有较多的历史交互、相似的文本兴趣、相似的活跃时间，则该用户更容易产生转发行为。
  - 现将各个方法的优缺点总结如下：
    - 基于用户过往行为的预测方法。
      - 假设用户的转发行为反映用户的兴趣，依据用户在预测时间点前的过往行为预测用户未来的行为。这类方法主要使用的模型是协同过滤模型，该模型能够挖掘用户兴趣，利用已知的用户偏好或行为预测未知的用户信息偏好或行为。但是由于微博信息时效性强，新信息不断产生，因此，此类方法面临较严重的新信息冷启动问题。融入用户属性特征、微博文本特征等可缓解冷启动问题。
    - 基于用户文本兴趣的预测方法。
      - 假设用户对某信息的转发行为主要源于用户对微博文本内容的兴趣，通过用户的过往微博文本信息对用户进行文本建模，从而预测用户对信息的转发行为。这类方法在用户拥有一定数量的微博文本信息时效果较好；但对于文本内容较少的用户，很难学到其真正感兴趣的内容。
    - 基于用户所受群体影响的预测方法。
      - 假设用户转发行为的产生源于所受群体的影响，包括信息发布者的影响和其他信息转发者的影响。这类方法中较多使用因子图模型，除用户之间的相互影响外，因子图模型还可建模其他影响因素，如内容流行度的影响等
    - 基于混合特征学习的预测方法。
      - 将转发行为预测视为二元分类问题，认为用户转发行为是多种因素作用的结果。分析影响用户转发行为的因素并将其表示为特征，然后选择适当的分类器训练分类模型。这种方法最为简单直观，模型解释性弱，依赖于特征的选择与组合。
    - 目前绝大部分的用户传播行为预测都是以静态网络拓扑结构、静态的用户行为为基础的，但是在现实中，微博信息传播速度极快，不断有新用户和新信息产生，无论是用户之间的关系网络，或是用户自身的行为和兴趣，都是随时间动态变化的。如何对用户传播行为进行动态建模是值得深入挖掘的问题。

##### **[计算图上的微积分：Backpropagation](http://chuansong.me/n/1779258)**

- Backpropagation (BP) 是使得训练深度模型在计算上可行的关键算法。对现代神经网络，这个算法相较于无脑的实现可以使梯度下降的训练速度提升千万倍。
- 除了在深度学习中的使用，BP 本身在其他的领域中也是一种强大的计算工具，例如从天气预报到分析数值的稳定性



































### 关于「当前系统项目」：**40%**

**1.目标明确**

：任务要明确，稍微改一点点，就是另一个目标了。参考自：[王露平博士](http://see.xidian.edu.cn/html/news/7897.html)

**1.1数据获取**

1.1.1工作：**理解**甲方提供Excel版：数据字典／样本数据：(订单明细、POP信息、POP商家评分、商品评论、商品基础信息等19张表格)，并**存储到mysql数据库中**：「数据导入」「数据库建表」等

1.1.2工具：[Linux基础 - get!](http://www.runoob.com/linux/linux-tutorial.html) 、 [sql基础 - get!](http://www.runoob.com/sql/sql-tutorial.html) 、   [mysql基础 - get!](http://www.runoob.com/mysql/mysql-tutorial.html)  

1.1.3参考：

1.2**数据初步分析**

1.2.1工作：原始19张表的数据分布初步探索「数据均值等分布」

1.2.2工具：[python基础 -- get！](http://www.runoob.com/python/python-tutorial.html) 、pandas

1.2.3参考：

**2.业务理解数据**

2.1**特征维度选择**

2.1.1工作：从原始19张表中结合业务理解，选择特征维度及特征列表

2.1.2工具：Hive。

2.1.3参考：

2.2**特征维度融合**

2.2.1工作：特征列表汇总到**核心目标**业务上的特征维度

2.2.2工具：Hive。

2.2.3参考：

**3.图谱构建**

3.1**图谱构建**

3.1.1工作：依据特征维度，构建图谱

3.1.2工具：Hive

3.1.3参考：

3.2图谱分析

3.2.1工作：基于图谱，提取特征

3.2.2工具：[python基础 -- get！](http://www.runoob.com/python/python-tutorial.html)、NetworkX

3.2.3参考：

**4.单模型特征应用**

4.1**回归模型选择**

4.1.1工作：单模型对提取出的特征进行评估

4.1.2工具：scikit-learn、XGB。

4.1.3参考：

4.2附：文本相似度分析过程：

4.2.1工作：匹配SELF与POP的名称

4.2.2工具：gensim、[python基础 -- get！](http://www.runoob.com/python/python-tutorial.html) 

4.2.3参考：

**5.多层级模型特征应用**

5.1**架构搭建**

5.1.1工作：多模型的架构搭建

5.1.2工具：

5.1.3参考：

-----

### 关于「基础技能」：

> 服务端：

[python基础 - get！](http://www.runoob.com/python/python-tutorial.html) 字符串、列表、字典、元组、IO、异常、多线程、面向对象、正则、MySQL、JSON、100实例

[Java基础 - get!](http://www.runoob.com/java/java-tutorial.html)    Java例子

[Linux基础 - get!](http://www.runoob.com/linux/linux-tutorial.html) 登陆、用户、文件、Vim

[Docker基础 - get!](http://www.runoob.com/docker/docker-tutorial.html)  容器引擎、虚拟化、打包镜像部署、沙盒、虚拟化

[PHP基础 - get！](http://www.runoob.com/php/php-tutorial.html)mysql、XML、AJAX

[正则表达式基础 - get!](http://www.runoob.com/regexp/regexp-tutorial.html)  

[JSP基础 - get!](http://www.runoob.com/jsp/jsp-tutorial.html)  

[Scala基础 - get!](http://www.runoob.com/scala/scala-tutorial.html)  

[设计模式基础 - get!](http://www.runoob.com/design-pattern/design-pattern-tutorial.html)

[Django基础 - get!](http://www.runoob.com/django/django-tutorial.html)  

[Servlet基础 - get!](http://www.runoob.com/servlet/servlet-tutorial.html)  

> 数据库：

[sql基础 - get!](http://www.runoob.com/sql/sql-tutorial.html) 查询、筛选、插入、更新、连接、合并、约束、删除、更改、函数

[mysql基础 - get!](http://www.runoob.com/mysql/mysql-tutorial.html) 匹配、索引、临时表、正则、去重、SQL注入

[MongoDB - get!](http://www.runoob.com/mongodb/mongodb-tutorial.html)  分布式文件存储

[Redis - get！](http://www.runoob.com/redis/redis-tutorial.html)  key-value存储系统

> 开发工具：

[git基础 - get!](http://www.runoob.com/git/git-tutorial.html)  分布式版本控制系统、克隆、修改、提交、工作区、版本库、分支、主线、合并、提交历史、标签、[github](http://www.runoob.com/w3cnote/git-guide.html)

> XML教程：

[XML基础 - get!](http://www.runoob.com/xml/xml-tutorial.html)  可扩展标记语言（e**X**tensible **M**arkup **L**anguage）、传输和存储数据

> Web Service:

[RSS基础 - get!](http://www.runoob.com/rss/rss-tutorial.html)    Really Simple Syndication（真正简易联合）

[RDF基础 - get!](http://www.runoob.com/rdf/rdf-intro.html)   RDF 是一个用于描述 Web 上的资源的框架

> HTML / CSS / JavaScript

[HTML基础 - get!](http://www.runoob.com/html/html-tutorial.html)

[CSS基础 - get!](http://www.runoob.com/css3/css3-tutorial.html)

[Boostrap基础 - get!](http://www.runoob.com/bootstrap/bootstrap-tutorial.html)

[JavaScript基础 - get!](http://www.runoob.com/js/js-tutorial.html)  

[jQuery基础 - get!](http://www.runoob.com/jquery/jquery-tutorial.html)

[AJAX基础 - get!](http://www.runoob.com/ajax/ajax-tutorial.html)  

[JSON基础 - get!](http://www.runoob.com/json/json-tutorial.html) 

[Highcharts基础 - get!](http://www.runoob.com/highcharts/highcharts-tutorial.html)  

> 网站建设：

[HTTP基础 - get!](http://www.runoob.com/http/http-tutorial.html)  HyperText Transfer Protocol、超文本传输协议、TCP/IP传输控制协议/因特网互联协议、客户端-服务端架构、C/S、通讯流程、消息结构、客户端请求消息、服务器响应消息、请求方法、HTTP状态码

[TCP/IP基础 - get!](TCP/IP基础 - get!)   TCP/IP 是因特网的通信协议、-**TCP 使用固定的连接、**IP 是无连接的、**IP 路由器**、**TCP/IP**、**IP 地址包含 4 组数字、**32 比特 = 4 字节、**IPV6**、**域名**、**TCP/IP 是不同的通信协议的大集合**

[网站主机基础 - get!](http://www.runoob.com/hosting/hosting-tutorial.html)  网站、web 服务器、ISP( Internet Service Provider ) Internet 服务提供商、每日的备份、流量限制、带宽或内容限制、域名是网站唯一的名称、主机解决方案中应包括域名注册、DNS 、月流量、POP 指的是邮局协议。、IMAP 指的是 Internet 消息访问协议。

[网站建设指南 - get!](http://www.runoob.com/web/web-buildingprimer.html)  World Wide Web（WWW)、信息存储文件：网页、web服务器、客户端、浏览器、获取网页：从服务器请求网页数据、HTTP请求包含网页地址、显示指令HTML、Hyper Text Markup Language（HTML）标记语言、<p>段落、Cascading Style Sheets（CSS）层叠样式表、样式表定义如何显示HTML元素、JavaScript客户端脚本、向HTML添加交互行为、EXtensible Markup Language（XML）可扩展标记语言、传输信息、ASP（Active Server Pages 动态服务器页面） 和 PHP（Hypertext Preprocessor技术，允许在网页中插入服务器可执行脚本） 、服务端脚本、动态改变内容、对HTML表单数据响应、访问数据、SQL、Web 创建：少即是多、导航一致、加载速度、用户反馈、显示器、Web标准：HTML、CSS、XML、XSL、DOM、Web语义化、语义网技术：描述语言和推理逻辑、语义网实现：XML及RDF （Resource Description Framework资源描述框架）、信息资源及其之间关系的描述：RDF、使用URI来标识不同的对象（包括资源节点、属性类或属性值）、可将不同的URI连接起来，清楚表达对象间的关系、**AdSense**、**AJAX (Asynchronous JavaScript and XML)**、**Apache**开源的Web服务器、**API (Application Programming Interface)**、**Browser**、**Client**、**Client/Server**、**Clickthrough Rate**、**Cloud Computing**、**Cookie**、**DB2**、**DBA (Data Base Administrator)**、**DNS (Domain Name Service)**计算机程序运行在Web服务器上域名翻译成IP地址、**DOS (Disk Operating System)**、**HTTP Client**计算机程序，从Web服务器请求服务、**HTTP Server**计算机程序，从Web服务器提供服务、**IIS (Internet Information Server)**适用于Windows操作系统的Web服务器、**IMAP (Internet Message Access Protocol)**电子邮件服务器检索电子邮件标准通信协议、**IP (Internet Protocol)**、**IP Address (Internet Protocol Address)**每一台计算机的一个独特的识别号码（如197.123.22.240）、**JSP (Java Server Pages)**基于Java技术允许在网页中插入服务器可执行的脚本、**LAN (Local Area Network)**局部地区（如建筑物内）的计算机之间的网络、**OS (Operating System)**、**Page Views**、**PDF (Portable Document Format)**、**Ping**、**Search Engine**、**TCP (Transmission Control Protocol)**、**TCP/IP (Transmission Control Protocol / Internet Protocol)**两台计算机之间的互联网通信协议的集合。 TCP协议是两台计算机之间的自由连接，而IP协议负责通过网络发送的数据包。、**URI (Uniform Resource Identifier)**用来确定在互联网上的资源。 URL是一种类型的URI。、**URL (Uniform Resource Locator)**Web地址。标准的办法来解决互联网上的网页文件（页）（如：http://www.w3cschool.cc/）、**VPN (Virtual Private Network)**两个远程站点之间的专用网络，通过一个安全加密的虚拟互联网连接（隧道）、**Web Services**软件组件和Web服务器上运行的应用程序、搜索引擎优化（Search Engine Optimization）SEO、提高一个网站在搜索引擎中的排名（能见度）的过程、百度搜索网站登录口、



--------

### 关于「Life」：

- ##### 礼貌的笑了笑，寸步不让

  > 老板，枉为人师！


- 别人一辈子都达不到的，才是价值的表彰

  > 翟墨，朗读 - 高尔基的《海燕》


- **结果**=**80%**(100%**想法**+100%**实施**)  

  > 打仗预留军，工作多思考；

  > **保留点实力，摸底很被动**；

  >主意跟我走，大家同向前。
  >

  - 有感于：[在职场里，看起来「尽全力」地工作是一件很蠢的事情吗](https://www.zhihu.com/question/60708921/answer/179744908?utm_medium=social&utm_source=wechat_session)  Jun 6

- > **「高考」光环已逝去，   「阶层」流动使命完。**

  > **「专业」紧随生产力，     未料想「已弃」高考。**
  >

  - 有感于：[高考40年，阶层分流的历史使命早就已经结束了](https://mp.weixin.qq.com/s?__biz=MzI0NzA3MTM5NQ%3D%3D&mid=2650557246&idx=1&sn=491c608e588acfdc6eaa05ddc4cccb9d#wechat_redirect) Jun 7




------
### 关于「基础知识」：如下：


#### 1.刷题

- 目的分析：基础知识的掌握牢固， 「 **基础熟练决定你的灵活性！」**
- 解决办法：大量做题，**「 不比别人熟练，必败」**
- 可能方式：**[纽克](https://www.nowcoder.com/7037691)、[柒越](https://www.julyedu.com)**

#### 2.简历

- 目的分析：让人接受你的努力， **「 被认可，被膜拜，这是主流！」**
- 时间节点：**6月底**

#### 3.经验

- 目的分析：系统性质的项目过程， **「 灵活的经验才能被主流朝拜！」**
- 解决办法：**实战**项目、开源项目， **「 片段化不系统，机会抓不住」**
- 可能方式：**[豪洞悉](http://geek.ai100.com.cn)、[艾科科](http://tinyletter.com/fly51fly/archive)、工种薅**、**英语**

---

### 关于「已完成项目／比赛」：

Dell EMC比赛：https://github.com/alare/Hackathon_2017

比赛硬件：树莓派， 虚拟机

比赛需要但未完成：Docker相关

比赛指南：[Introduction](https://github.com/alare/Hackathon_2017/blob/master/documentation/Mars-challenge-instructions.md)

比赛得分：[PointsGet](https://github.com/alare/Hackathon_2017/blob/master/documentation/Mars-challenge-points-table.md)

1.第一部分：传感器获取数据

1.1.工作：后台运行多个传感器go文件并集成。并在浏览器查看传感器的传入数据。

1.2.工具：后台运行go命令：`nohup go run flare.go &` 。端口如下：0.0.0.0:9000（localhost）。环境变量env

1.3.参考：[Setting up the Raspberry Pi Sensors](https://github.com/alare/Hackathon_2017/blob/master/documentation/Raspberry-Go-Weather-Simulator-Setup.md)   以及  [SensorSuite](https://github.com/alare/Hackathon_2017/tree/master/sensorsuite) 

2.第二部分：平台策略执行

2.1工作：在浏览器显示出监控数据，需要运行docker

2.2工具：[Game Controller](https://github.com/alare/Hackathon_2017/tree/master/game-controller) 以及[Dashboard](https://github.com/alare/Hackathon_2017/tree/master/dashboard) 以及 [Testing the Command and Control Center](https://github.com/alare/Hackathon_2017/blob/master/documentation/Mars-challenge-instructions.md#testing-the-command-and-control-center)

3.第三部分：各个队伍PK

3.1工作：更改策略，使得最后存活时间更长，注意更改url以进行队伍pk

3.2工具：[Team_Strategy](https://github.com/alare/Hackathon_2017/tree/master/clients) 。[Testing the Command and Control Center](https://github.com/alare/Hackathon_2017/blob/master/documentation/Mars-challenge-instructions.md#testing-the-command-and-control-center)

比赛过程：**「晚上10点的没有的微信」**— 那时候，很辛苦

 ![going](YangQiDaily/going.jpg)

比赛结果：「运气比较好，挑战了清华、北大」

 ![Hackathon](YangQiDaily/Hackathon.jpg) 



![FirstPrize](YangQiDaily/FirstPrize.jpg)

