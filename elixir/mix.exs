defmodule ElixirImageRecognizer.MixProject do
  use Mix.Project

  def project do
    [
      app: :elixir_image_recognizer,
      version: "0.1.0",
      elixir: "~> 1.20",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.10"},
      {:exla, "~> 0.10"},
      {:axon, "~> 0.7"},
      {:scidata, "~> 0.1"}
    ]
  end
end
