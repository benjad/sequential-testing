import pytest
from sequential_testing.sprt import SPRTBinomial, SPRTNormal


class TestSPRTBinomial:
    @pytest.fixture
    def binomial(self):
        return SPRTBinomial(alpha=0.05, beta=0.2, h0=0, h1=0.1)

    def test_hipothesis_evaluation_1(self, binomial):
        binomial.a = -1
        binomial.b = 1
        binomial._hipothesis_evaluation(0, 1)
        assert len(binomial.h0_acepted) == 0 and len(binomial.h0_rejected) == 0

    def test_hipothesis_evaluation_2(self, binomial):
        binomial.a = -1
        binomial.b = 1
        binomial._hipothesis_evaluation(2, 1)
        assert len(binomial.h0_acepted) == 0 and len(binomial.h0_rejected) == 1

    def test_hipothesis_evaluation_3(self, binomial):
        binomial.a = -1
        binomial.b = 1
        binomial._hipothesis_evaluation(-2, 1)
        assert len(binomial.h0_acepted) == 1 and len(binomial.h0_rejected) == 0    


class TestSPRTNormal:
    @pytest.fixture
    def normal(self):
        return SPRTNormal(alpha=0.05, beta=0.2, sigma=1, h0=0, h1=0.1)

    def test_hipothesis_evaluation_1(self, normal):
        normal.a = -1
        normal.b = 1
        normal._hipothesis_evaluation(0, 1)
        assert len(normal.h0_acepted) == 0 and len(normal.h0_rejected) == 0

    def test_hipothesis_evaluation_2(self, normal):
        normal.a = -1
        normal.b = 1
        normal._hipothesis_evaluation(2, 1)
        assert len(normal.h0_acepted) == 0 and len(normal.h0_rejected) == 1

    def test_hipothesis_evaluation_3(self, normal):
        normal.a = -1
        normal.b = 1
        normal._hipothesis_evaluation(-2, 1)
        assert len(normal.h0_acepted) == 1 and len(normal.h0_rejected) == 0   
