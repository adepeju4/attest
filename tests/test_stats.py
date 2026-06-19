from attest.scoring.stats import wilson_interval, difference_is_real, two_proportion_z


def test_wilson_basic_bounds():
    p = wilson_interval(41, 50)
    assert 0.0 <= p.low <= p.rate <= p.high <= 1.0
    assert p.rate == 41 / 50
    assert p.n == 50


def test_wilson_handles_zero_and_full():
    assert wilson_interval(0, 10).low == 0.0
    assert wilson_interval(0, 10).high > 0.0
    assert wilson_interval(10, 10).high == 1.0
    assert wilson_interval(10, 10).low < 1.0


def test_wilson_empty():
    p = wilson_interval(0, 0)
    assert (p.rate, p.low, p.high, p.n) == (0.0, 0.0, 0.0, 0)


def test_smaller_n_gives_wider_interval():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_difference_is_real():
    assert difference_is_real(90, 100, 60, 100) is True
    assert difference_is_real(52, 100, 48, 100) is False


def test_two_proportion_z_sign():
    assert two_proportion_z(90, 100, 60, 100) > 0
    assert two_proportion_z(60, 100, 90, 100) < 0
