package neural

import "testing"

func TestSigmoidLayer(t *testing.T) {
	var layerWeights = [][]float64{
		{0.2, 0.3},
		{0, 0.5},
		{2.1, 0},
	}

	var layerBias = []float64{
		0.6,
		0.7,
		-0.3,
	}
	testLayer, err := makeSigmoidLayer(layerWeights, layerBias, 2)
	if err != nil {
		t.Errorf("makeWeightLayer returned unexpected error")
	}
	_ = testLayer
}

func TestSigmoidNetword(t *testing.T) {
	var networkWeights = [][][]float64{
		{
			{0.2, 0.3},
			{0, 0.5},
			{2.1, 0},
		},
		{
			{1.0, 2.0, 3.0},
			{0, 1, 0},
		},
		{
			{1, 1},
		},
	}

	var networkBias = [][]float64{
		{0.5, 0.2, 0.1},
		{2.2, -0.2},
		{0.5},
	}
	testNet := MakeSigmoidNetwork(2, networkWeights, networkBias)
	_ = testNet
}

func TestSmallSigmoidNet(t *testing.T) {
	var smallNetWeights = [][][]float64{
		{
			{1, 1},
			{1, -1},
		},
		{
			{1, 1},
		},
	}

	var smallNetBias = [][]float64{
		{0.3, -0, 4},
		{0.5},
	}
	smallNet := MakeSigmoidNetwork(2, smallNetWeights, smallNetBias)
	smallNet.initInput()
	/*
		    3   3.5     1.5
			4   -0.5
	*/
	{
		smallNet.SetInput([]float64{3, 4})
		smallNet.EvaluateNetwork()
		actual := smallNet.GetOutputValue()
		expect := 0.8542419408390483
		if actual != expect {
			t.Errorf("smallNet got %v, expected %v\n weights:\n^v",
				actual, expect, smallNet)
		}
	}
	/*
		    2   2     1
			2   0
	*/
	{
		smallNet.SetInput([]float64{2, 2})
		smallNet.EvaluateNetwork()
		actual := smallNet.GetOutputValue()
		expect := 0.8793843569485015
		if actual != expect {
			t.Errorf("smallNet got %v, expected %v\n weights:\n^v",
				actual, expect, smallNet)
		}
	}
}

func TestRandomSigmoid(t *testing.T) {
	net, weights, biases := MakeRandomSigmoid([]int{10, 4, 2})
	if len(weights) == 0 || len(weights) != len(biases) {
		t.Errorf("Weights and biases incorrect")
	}
	net.SetInput([]float64{0, 0, 0.1, 0.4, 0, 0, -3, 0.3, 0, 0})
	net.EvaluateNetwork()
}
