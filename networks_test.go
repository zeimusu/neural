package neural

import (
	"fmt"
	"reflect"
	"testing"
)

var net = Network{
	inputLayer,
	twoSumsLayer,
	sumAvLayer,
	outputLayer,
}

func TestEvaluateNetwork(t *testing.T) {
	net.EvaluateNetwork()
	/*
		0.3    1.5    3.0    4.5
		0.5    1.5    1.5
		0.7
	*/
	if net.GetOutputValue() != 4.5 {
		for _, l := range net {
			fmt.Println(getValues(l))
		}
		t.Errorf("Expected 4.5, got %v", net.GetOutputValue())
	}
}

func TestRandomNetwork(t *testing.T) {
	n, _ := MakeRandomSumNetwork([]int{2, 4, 4, 1})
	if len(n) != 4 {
		t.Errorf("Network has wrong length, expected 4, got %v", len(n))
	}
	if len(n[0]) != 2 {
		t.Errorf("input layer has length %v, expected 2", len(n[0]))
	}
	if len(n[1]) != 4 || len(n[2]) != 4 {
		t.Errorf("process layer has length %v,%v, expected 4,4", len(n[1]), len(n[2]))
	}
	if len(n[3]) != 1 {
		t.Errorf("output layer has length %v, expected 1", len(n[0]))
	}

	inputvalues := getValues(n[0])
	if !reflect.DeepEqual(inputvalues, []float64{0, 0}) {
		t.Errorf("Inputlayar not initialised to zero")
	}
	n.SetInput([]float64{0, 0})
	n.EvaluateNetwork()
	if n.GetOutputValue() != 0 {
		for _, l := range net {
			fmt.Println(getValues(l))
		}
		t.Errorf("Expected 0 output with  0 input, got %v", n.GetOutputValue())
	}
	/*
		n.SetInput([]float64{2, 2})
		n.EvaluateNetwork()
		if n.GetOutputValue() != 2 {
			for _, l := range n {
				fmt.Println(getValues(l))
			}
			t.Errorf("Expected 2 output with  2,2 input, got %v", n.GetOutputValue())
		}

		n.SetInput([]float64{4, 5})
		n.EvaluateNetwork()
		for _, l := range n {
			fmt.Println(getValues(l))
		}
		t.Log("Got %v with input 4,5", n.GetOutputValue())
	*/
}
