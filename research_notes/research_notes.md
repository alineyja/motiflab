В базовом классе Alphabet есть абстрактный метод complement_symbol.
Для ДНК и РНК это логично. Но когда мы добавим ProteinAlphabet, у аминокислот нет понятия "комплементарности". Нам придется писать метод-заглушку, который выбрасывает NotImplementedError.
Как это решают в больших библиотеках: Делают NucleotideAlphabet(Alphabet), в котором появляется complement, а ProteinAlphabet наследуется напрямую от Alphabet.

1D Convolutional Neural Network for DNA Motif Discovery.
    
    Architecture:
    - Conv1D: Scans DNA sequence for patterns.
    - ReLU: Activation.
    - AdaptiveMaxPool1d: Extracts the maximum activation signal across the sequence (Global Max Pooling).
    - Linear: Final classification layer.
    Forward pass.
        Args:
            x: Tensor of shape (Batch, 4, Length)
        Returns:
            Logits tensor of shape (Batch, 1). 
            Note: We return logits (no Sigmoid here) for numerical stability with BCEWithLogitsLoss.
    Method A: Direct Softmax of Conv1D weights.
        
        Args:
            weights: Weights of shape (num_filters, 4, kernel_size)
        Returns:
            Position Probability Matrices (PPM) of shape (num_filters, 4, kernel_size)
            where each column sums to 1.0.
    Method B: Extract sub-sequences (seqlets) from real DNA sequences 
        that cause high activation (> threshold * max_activation) of a specific filter.
        
        Args:
            model: Trained MotifCNN model.
            sequences: List of DNASequence objects to scan.
            filter_idx: Index of the Conv1D filter to analyze.
            threshold: Fraction of maximum activation to consider as a "hit".
        Returns:
            List of string seqlets of length `kernel_size`.

$ python benchmarks/bench_encoder.py
генерация днк длиной 1,000,000 bp
запуск бенчмарка (10 иттераций)
   NumPy среднее время 0.08313 seconds
  C++  среднее время 0.01287 seconds
BENCHMARK RESULTS (Sequence length: 1,000,000 bp)
NumPy average time: 83.13 ms
C++ average time:   12.87 ms
 C++ Encoder is 6.46x faster than NumPy


        proof of principle:
     Генерация  генома пустышки создадим файл mock_genome.fa с одной хромосомой chr_mock длиной 150 000 нуклеотидов.
    Имплантация «секретного» мотива выберем 1000 случайных участков по 100 нуклеотидов. В каждый из них в случайное место вживим мотив ATGCGATC (наш синтетический CTCF). Координаты этих участков запишем в mock_peaks.bed.
    Обучение нейросети обучим нашу модель MotifCNN на этих данных. Поскольку задача простая, сеть должна за пару эпох выйти на точность >95%.
    Декодирование «секрета» (Интерпретация):
    найдем самый важный фильтр нейросети (тот, у которого самый большой вес в полносвязном классификаторе).

    С помощью MotifInterpreter мы извлечем его PPM-матрицу.

    восстановим консенсусную последовательность этого фильтра и сравним её с оригинальным вживленным мотивом ATGCGATC (или его обратным комплементом GATCGCAT).

    $ python examples/ctcf_chr22_train.py
парсинг BED-файла CTCF
   Общее число пиков 47,985
   пики на хромосоме 22 906
инициализация экстрактора ДНК и датасета из FASTA

Training on: cpu

 обучение модели
Epoch 1/10 | Train Loss: 0.6952 | Val Accuracy: 50.69%
Epoch 2/10 | Train Loss: 0.6874 | Val Accuracy: 80.99%
Epoch 3/10 | Train Loss: 0.6683 | Val Accuracy: 78.24%
Epoch 4/10 | Train Loss: 0.6570 | Val Accuracy: 84.02%
Epoch 5/10 | Train Loss: 0.6398 | Val Accuracy: 90.36%
Epoch 6/10 | Train Loss: 0.6257 | Val Accuracy: 90.91%
Epoch 7/10 | Train Loss: 0.6050 | Val Accuracy: 93.11%
Epoch 8/10 | Train Loss: 0.5856 | Val Accuracy: 92.56%
Epoch 9/10 | Train Loss: 0.5613 | Val Accuracy: 93.66%
Epoch 10/10 | Train Loss: 0.5356 | Val Accuracy: 92.29%

 извлечение мотивов из обученной модели
Active filter: Index 8 (Linear weight: 0.2866)
REAL BIOLOGY EXPERIMENT RESULT
CTCF Motif Consensus (JASPAR): CCACYAGGTGGCAG
Reconstructed by MotifCNN:     TTGCCACCTGGTGGC
Filter PPM Matrix (Probabilities):
Pos:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
  A:  24  22  23  23  23  28  22  23  27  24  23  24  22  23  24
  C:  25  25  24  30  31  25  29  30  23  23  25  22  23  23  28
  G:  25  26  29  24  22  22  25  22  22  29  28  25  32  31  23
  T:  26  27  24  23  24  24  24  24  29  25  23  28  23  23  25
Information Content sum (bits): 0.11 / 30.0
Performs in silico mutagenesis to evaluate the functional impact 
    of every possible single-nucleotide mutation in a DNA sequence.