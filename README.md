Custom Interpretable DL Framework for Regulatory DNA Analysis

Follow these steps to set up the development environment from scratch
1. Clone the Repository
```bash
git clone https://github.com/alineyja/motiflab.git
cd motiflab
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# linux/MacOs
python3 -m venv .venv
source .venv/bin/activate
```
3. Install Dependencies
```
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy pytest pyfaidx pybind11 tqdm
#choose the installation command matching your hardware. For CUDA-enabled training (recommended)
# Example for CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
4. Building C++ / CUDA Extensions
Ensure you have a C++ compiler installed (e.g., MSVC via Visual Studio Build Tools or GCC on Linux).
```
python setup.py build_ext --inplace
python setup_cuda.py build_ext --inplace
```
5. Download Genomic Data
```
python scripts/download_data.py
python scripts/import_bed_to_sql.py
python examples/train_regression.py
python examples/genome_scanner.py
```
Project Goal

The goal of the project is to automatically detect regulatory DNA motifs directly from raw genomic sequences, using the genome-architectural protein CTCF (CCCTC-binding factor) as a case study, and to interpret and visualize the discovered motifs in both 2D and 3D.

Algorithm and Pipeline
Genomic coordinate parsing — reads ChIP-seq peak files in BED/narrowPeak format in streaming mode.
Efficient DNA extraction — rapidly retrieves the required genomic regions from reference genomes in FASTA format without loading the entire genome into memory.
Reverse-complement processing — automatically performs a reverse-complement transformation when genomic coordinates correspond to the negative DNA strand.
Dynamic nucleotide shuffling — dynamically shuffles nucleotides to balance the training data and generate appropriate negative examples.
One-hot encoding — converts DNA sequences represented as A, C, G, and T into numerical tensors suitable for neural network processing.
MotifCNN — a custom 1D convolutional neural network scans DNA sequences for sequence patterns. A Global Max Pooling layer extracts the strongest activation regardless of the motif's position within the input window.
Hardware optimization — the model is designed to run efficiently on both CPUs and modern NVIDIA GPUs, including the Blackwell architecture (sm_120).
Filter-weight decoding — convolutional filter weights are transformed into classical Position Weight Matrices (PWMs), allowing abstract neural-network parameters to be interpreted as biologically meaningful nucleotide motifs. Information content is also calculated for each position.
In vivo genomic scanning — the trained model scans real genomic DNA to identify regions responsible for the strongest network activations. These sequences are then used to construct a frequency matrix representing the discovered motif.
Sequence Logo generation — generates visual motif representations in which the height of each nucleotide reflects its importance and conservation at the corresponding position.
In Silico Mutagenesis — evaluates the functional contribution of individual nucleotide positions. Each nucleotide in a peak is replaced with the three alternative nucleotides, and the resulting changes in the predicted binding probability are visualized as heatmaps.
3D protein visualization — retrieves an experimentally determined protein structure from the Protein Data Bank (PDB) and generates an interactive HTML widget in which the user can rotate and examine the 3D structure of the protein bound to DNA.
Custom CUDA Conv1D Kernel — a custom CUDA kernel was implemented from scratch to perform convolution in parallel across thousands of GPU threads. The kernel is compiled for the latest NVIDIA Blackwell architecture (sm_120) using PTX intermediate code generation.
Experimental Results

An experiment was conducted in which the model was trained exclusively on chromosome 21 (chr21) and evaluated on a completely independent chromosome, chromosome 22 (chr22), which was not seen during training.

The model achieved an accuracy of 93.39% on chr22.

The convolutional filters successfully reconstructed a CTCF-related sequence motif:

CCACCTGGTGGC

The extracted motif matches the corresponding motif in the JASPAR database with nucleotide-level agreement.

The in silico mutagenesis heatmaps demonstrated that the model effectively ignores random genomic noise at the edges of ChIP-seq peaks while showing a strong response to mutations within conserved regions of the motif. Mutations in critical nucleotide positions, particularly within conserved GC-rich regions, resulted in a substantial decrease in the predicted binding probability.

Project Development

The project is focused on the computational analysis and discovery of sequence motifs in DNA, RNA, and protein sequences.

As of August 1, 2026, Core 1.0 has been completed. The current version includes:

a neural network for sequence classification;
a custom One-Hot encoder;
support for genomic data in FASTA format;
benchmarking of the custom encoder against an existing NumPy-based implementation;
automatic extraction and interpretation of convolutional filters;
conversion of learned convolutional weights into PWMs;
motif visualization and analysis.

During training, the convolutional filters are learned automatically through backpropagation and gradient-based optimization. After training, the learned filters are converted into PWMs, making it possible to interpret the sequence patterns discovered by the neural network.

Planned and Implemented Extensions

The project is being developed toward a more general-purpose framework for motif discovery in biological sequences.

Planned and ongoing components include:

Multithreading — implemented as of August 2, 2026;
CUDA acceleration;
a custom CNN implementation specifically designed for motif discovery in DNA, RNA, and protein sequences;
integration of the custom-autograd-C project or development of a custom automatic differentiation engine;
implementation of custom optimization algorithms;
implementation of activation functions such as Sigmoid and Softmax;
implementation of loss functions including Binary Cross-Entropy and Log Loss.

The long-term objective is to transform the project from a specialized CTCF motif-detection model into a general-purpose, interpretable computational framework for discovering and analyzing sequence motifs across DNA, RNA, and protein sequences.

Results & Validation
Blind Test Accuracy: Achieved 90.9% classification accuracy and R=0.845 Pearson correlation on unseen human Chromosome 22.
Motif Recovery: Successfully reconstructed the core CTCF binding motif (CCACCTGG...) matching the JASPAR database standard.

Цель проекта автоматическое обнаружение регуляторных мотивов ДНК (на примере белка-архитектора генома CTCF) непосредственно из сырых геномных последовательностей,а также их интерпретация и визуализация в 2D и 3D.
алгоритм:
парсинг геномных координат читает файлы пиков ChIP-seq (формат BED/narrowPeak) в потоковом режиме.
быстро извлекает нужные участки ДНК из референсных геномов (FASTA) без загрузки всего файла в память.Выполняет операцию обратного комплемента (reverse_complement), если координаты лежат на минус цепи.
для балансировки обучения программа динамически перемешивает нуклеотиды.
one-hot кодирование: Переводит текстовые строки ДНК (A, C, G, T) в математические тензоры.
MotifCNN 1D-свертка сканирует ДНК на наличие паттернов, а слой Global MaxPooling вычленяет сильнейший сигнал связывания независимо от его положения в окне.
Модель оптимизирована для обучения как на CPU, так и на новейших графических процессорах NVIDIA (архитектура Blackwell, sm_120).
Декодирование весов фильтра (PWM) превращает абстрактные веса нейронов в классические биологические матрицы вероятностей нуклеотидов с расчетом информационной ценности.
Сканирование живой ДНК находит реальные участки генома, вызвавшие максимальный отклик сети, и строит по ним частотную матрицу
Sequence Logo генерирует графики логотипов мотивов, где размер буквы отражает важность позиции для связывания белка.
in Silico Мутагенез heatmaps  позволяет оценить функциональный вклад каждой позиции. программа заменяет каждый нуклеотид в пике на три альтернативных и строит карту влияния мутаций на вероятность связывания.
3D-рендеринг скачивает экспериментально доказанную структуру белка из PDB и генерируе html виджет,где можно крутить модель белка, обхватывающего спираль ДНК.
Custom CUDA Conv1D Kernel  написанный с нуля CUDA-кернел выполняющий параллельную свертку на тысячах ядер. Скомпилирован под новейшую архитектуру Blackwell (sm_120) с использованием технологии генерации промежуточного кода PTX.
проведен  эксперимент обучил модель на хромосоме chr21, а протестировал на полностью независимой, не виденной сетью хромосоме chr22.
точность на chr22 93.39%
реконструкция CTCF мотива модель успешно извлекла из весов ядро-паттерн CTCF CCACCTGGTGGC, который с точностью до буквы совпадает с базой данных JASPAR.
heatmap'ы мутаций доказали,что модель безошибочно игнорирует случайный геномный шум по краям пика и остро реагирует на мутации в консервативных ядрах мотива (гуаниновые и цитозиновые повторы), падением вероятности связывания.\



проект направленный на исследование мотивов в последовательностях ДНК, РНК и белков. На данный момент(01.08.26) готова версия core 1.0 включающая себя нейронную сеть, собственный one hot encoder, с использованием данных FASTA и сравнение скорости работы собственного encoder'а с уже готовым из numpy.
после обучения сети веса сверточных фильтров преобразуются в PWM для интерпретации найденных мотивов, сеть самостоятельно обучает веса сверточных фильтров порсредством обучения от обратного распростронения ошибок
У проекта планируется добавление мултипоточности(сделано на момент 02.08.26), cuda, так же собственной реализации самостоятельной cnn сети специализированной для поиска мотивов в последовательностях ДНК, РНК и белков, так же либо интегрирование(https://github.com/alineyja/custom-autograd-C) либо же реализация собственной системы автоматического дифференцирования, алгоритмов оптимизации,функций активации Sigmoid, Softmax и функций потерь Binary Cross-Entropy, Log Loss.

