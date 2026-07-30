import pytest
from sequential_testing.msprt import MSPRTBinomial, MSPRTNormal


class TestMSPRTBinomial:
    @pytest.fixture
    def binomial(self):
        return MSPRTBinomial(alpha=0.05, beta=0.2, sigma=1, tau=None, h0=0, m=100)

    def test_hipothesis_evaluation_1(self, binomial):
        binomial.b = 1
        binomial._hipothesis_evaluation(0, 1)
        assert len(binomial.h0_acepted) == 0 and len(binomial.h0_rejected) == 0

    def test_hipothesis_evaluation_2(self, binomial):
        binomial.b = 1
        binomial._hipothesis_evaluation(2, 1)
        assert len(binomial.h0_acepted) == 0 and len(binomial.h0_rejected) == 1

    def test_hipothesis_evaluation_3(self, binomial):
        binomial.b = 1
        binomial._hipothesis_evaluation(-2, 1)
        assert len(binomial.h0_acepted) == 0 and len(binomial.h0_rejected) == 0    


class TestMSPRTNormal:
    @pytest.fixture
    def normal(self):
        return MSPRTNormal(alpha=0.05, beta=0.2, sigma=1, tau=None, h0=0, m=100)

    def test_hipothesis_evaluation_1(self, normal):
        normal.b = 1
        normal._hipothesis_evaluation(0, 1)
        assert len(normal.h0_acepted) == 0 and len(normal.h0_rejected) == 0

    def test_hipothesis_evaluation_2(self, normal):
        normal.b = 1
        normal._hipothesis_evaluation(2, 1)
        assert len(normal.h0_acepted) == 0 and len(normal.h0_rejected) == 1

    def test_hipothesis_evaluation_3(self, normal):
        normal.b = 1
        normal._hipothesis_evaluation(-2, 1)
        assert len(normal.h0_acepted) == 0 and len(normal.h0_rejected) == 0   
