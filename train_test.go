package neural

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func makeRandInputs() ([][]float64, [][]float64) {
	inputs := make([][]float64, 40)
	desired := make([][]float64, 40)
	for i := range inputs {
		inputs[i] = make([]float64, 10)
		for j := range inputs[i] {
			if (i+j)%2 == 0 {
				inputs[i][j] = rand.Float64()
			} else {
				inputs[i][j] = 0
			}
		}
		if i%2 == 0 {
			desired[i] = []float64{1, 0}
		} else {
			desired[i] = []float64{0, 1}
		}
	}
	return inputs, desired
}

func TestShuffle(t *testing.T) {
	inputs, desired := makeRandInputs()
	si, sd := shuffleInputsDesired(inputs, desired)
	if len(si) != len(inputs) {
		t.Errorf("Shuffling changes length inputs")
	}
	if len(sd) != len(desired) {
		t.Errorf("Shuffling changes length inputs")
	}
	if reflect.DeepEqual(si, inputs) {
		t.Errorf("Inputs and shuffled equal")
	}
	for i := range si {
		if si[i][0] == 0 && sd[i][0] == 1 {
			t.Errorf("Shuffling alignment broken")
		}
		if si[i][0] != 0 && si[i][0] == 0 {
			t.Errorf("Shuffling alignment broken")
		}
	}
}

func TestDcdw(t *testing.T) {
	inputs, desired := makeRandInputs()
	_, weights, biases := MakeRandomSigmoid([]int{10, 4, 2})
	fmt.Println("testdcdw")
	gradient := dCdw(0, 2, 4, inputs[0:10], desired[0:10], weights, biases)
	fmt.Println("grad=", gradient)
	gradient = dCdw(0, 2, 4, inputs[10:20], desired[10:20], weights, biases)
	fmt.Println("grad=", gradient)
}
func TestDcdb(t *testing.T) {
	inputs, desired := makeRandInputs()
	_, weights, biases := MakeRandomSigmoid([]int{10, 4, 2})
	fmt.Println("testdcdb")
	gradient := dCdb(0, 3, inputs[0:10], desired[0:10], weights, biases)
	fmt.Println("grad=", gradient)
}

func TestTrain(t *testing.T) {
	net, weights, biases := MakeRandomSigmoid([]int{10, 4, 2})
	inputs, desired := makeRandInputs()
	initialCost := TotalCost(net, inputs, desired)
	newWeights, newBiases := TrainSig(inputs, desired, weights, biases)
	fmt.Println("weights", weights)
	fmt.Println("biase", biases)
	fmt.Println("neweights", newWeights)
	fmt.Println("newbiases", newBiases)
	newnet := MakeSigmoidNetwork(10, newWeights, newBiases)
	postCost := TotalCost(newnet, inputs, desired)
	fmt.Println("costs", initialCost, postCost)
}

func TestIterateTrain(*testing.T) {
	net, weights, biases := MakeRandomSigmoid([]int{10, 4, 2})
	inputs, desired := makeRandInputs()
	epochs := 30
	var cost float64
	for i := 0; i < epochs; i++ {
		cost = TotalCost(net, inputs, desired)
		fmt.Println(cost)
		newW, newB := TrainSig(inputs, desired, weights, biases)
		net = MakeSigmoidNetwork(10, newW, newB)
	}
	cost = TotalCost(net, inputs, desired)
	fmt.Println("final cost", cost)
}

/*
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
	net.SetInput([]float64{0, 0, 0.1, 0.4, 0, 0, -3, 0.3, 0, 0})
	net.EvaluateNetwork()
}
*/
