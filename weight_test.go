package main

import "testing"

var layerWeights = [][]float64{
	{0.2, 0.3},
	{0, 0.5},
	{2.1, 0},
}

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

var smallNetWeights = [][][]float64{
	{
		{1, 1},
		{1, -1},
	},
	{
		{1, 1},
	},
}

func TestWeightLayer(t *testing.T) {
	testLayer, err := makeWeightLayer(layerWeights, 2)
	if err != nil {
		t.Errorf("makeWeightLayer returned unexpected error")
	}
	_ = testLayer
}

func TestWeightNetword(t *testing.T) {
	testNet := MakeWeightedNetwork(2, networkWeights)
	_ = testNet
}

func TestSmallNet(t *testing.T) {
	smallNet := MakeWeightedNetwork(2, smallNetWeights)
	smallNet.initInput()
	/*
		    3   3.5     1.5
			4   -0.5
	*/
	{
		smallNet.SetInput([]float64{3, 4})
		smallNet.EvaluateNetwork()
		actual := smallNet.GetOutputValue()
		expect := 1.5
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
		expect := 1.
		if actual != expect {
			t.Errorf("smallNet got %v, expected %v\n weights:\n^v",
				actual, expect, smallNet)
		}
	}
}
