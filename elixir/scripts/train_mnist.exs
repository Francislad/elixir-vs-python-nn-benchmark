defmodule MnistTrain do
  def run do
    IO.puts("Downloading MNIST dataset (Scidata)...")

    {{train_images_bin, train_type, train_shape}, {train_labels_bin, _, _}} =
      Scidata.MNIST.download()

    {{test_images_bin, _, test_shape}, {test_labels_bin, _, _}} =
      Scidata.MNIST.download_test()

    train_n = elem(train_shape, 0)
    test_n = elem(test_shape, 0)

    train_images =
      train_images_bin
      |> Nx.from_binary(train_type)
      |> Nx.reshape({train_n, 784})
      |> Nx.divide(255.0)

    test_images =
      test_images_bin
      |> Nx.from_binary(train_type)
      |> Nx.reshape({test_n, 784})
      |> Nx.divide(255.0)

    train_labels =
      train_labels_bin
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({train_n, 1})
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

    test_labels =
      test_labels_bin
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({test_n, 1})
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

    batch_size = 128

    train_data =
      Stream.zip(
        Nx.to_batched(train_images, batch_size),
        Nx.to_batched(train_labels, batch_size)
      )

    model =
      Axon.input("input", shape: {nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dropout(rate: 0.5)
      |> Axon.dense(10, activation: :softmax)

    IO.puts("Starting MNIST training (GPU via EXLA)...")
    start_time = System.monotonic_time(:millisecond)

    model_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(train_data, %{}, epochs: 5, compiler: EXLA)

    elapsed_ms = System.monotonic_time(:millisecond) - start_time
    IO.puts("Training time MNIST: #{elapsed_ms / 1000} seconds")

    test_data = Stream.zip(Nx.to_batched(test_images, batch_size), Nx.to_batched(test_labels, batch_size))

    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(test_data, model_state, compiler: EXLA)
  end
end

MnistTrain.run()
