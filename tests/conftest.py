import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture
def dim16():
    return 16


@pytest.fixture
def dim64():
    return 64


@pytest.fixture
def rng_seed():
    torch.manual_seed(42)
    yield


@pytest.fixture(scope="module")
def toy_bilevel_problem():
    def inner_objective(w, lam):
        return (1.0 - w) ** 2 + 100.0 * (w ** 2 - lam) ** 2

    def outer_objective(lam):
        w = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            loss = inner_objective(w, lam.detach())
            loss.backward()
            opt.step()
        return (w.detach() - 1.0) ** 2

    return inner_objective, outer_objective
