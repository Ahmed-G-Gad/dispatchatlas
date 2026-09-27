"""Population metaheuristics on priority vectors.

Every method here keeps a population of real vectors in the unit hypercube and
moves it with the update rule of its paper. A vector holds one key per task,
plus, by default, one per task with several modes, and decodes through
:class:`~dispatchatlas.solvers.encoding.VectorEncoding` into a task order and
mode choices, which the shared schedule builder turns into a schedule. The
engine in :mod:`~dispatchatlas.solvers.swarm._engine` does everything the
methods share: the uniformly random initial population, bounds, evaluation,
the budget schedule, memories of the best vectors, and numerical helpers. Each
method module implements one iteration of its rule, and its docstring states
the rule with the paper's equation numbers and where the port has to choose
between the paper and the authors' implementation.

**Representation.** The ``rule`` parameter of every method selects how keys
become an order: ``"random-key"``, the default, sorts tasks by ascending key
(Bean 1994), and ``"largest-value"`` by descending key, the largest-order-value
rule of many swarm scheduling papers. Both reach every order, but they differ
for update rules that are not symmetric under ``x -> 1 - x``, such as rules
drawn towards the origin. The ``modes`` parameter selects where the modes of
tasks with several modes come from. With ``"keys"``, the default, one more key
per such task selects its mode, so the search covers every mode choice,
including slower modes that save energy or cost. With ``"builder"`` the vectors
hold task keys only and the schedule builder gives each task its
earliest-finishing mode. This suits time-based objectives such as the makespan
or tardiness, where learning modes from keys can leave these methods behind
random search, which lets the builder choose modes too.

**Budget.** Each decoded vector costs one evaluation of the run's budget, the
initial population included, and one iteration applies the update rule to the
whole population. Most methods evaluate one vector per member and iteration;
teaching-learning-based optimization and the marine predators algorithm
evaluate two, the artificial bee colony two plus one per restart, cuckoo search
one plus one per replaced member, and Harris hawks optimization one or two. The
rules that the papers schedule over an iteration budget use the fraction of
the actual budget spent, so every method honours any single bounded stop
criterion.

**Methods.** ``grey-wolf-optimizer``, ``whale-optimization-algorithm``,
``firefly-algorithm``, ``moth-flame-optimization``, ``sine-cosine-algorithm``,
``arithmetic-optimization-algorithm``, ``harris-hawks-optimization``,
``teaching-learning-based-optimization``, ``salp-swarm-algorithm``,
``marine-predators-algorithm``, ``cuckoo-search``, ``artificial-bee-colony``,
``gravitational-search-algorithm``, and ``jaya``.

**Caveats.** These methods are provided as reproducible baselines, not as
recommendations. Sörensen (2015) argues that metaphor-based methods hide
familiar mechanisms behind new vocabulary and are rarely validated with rigour.
Camacho-Villalón et al. show that the grey wolf, firefly, moth-flame and whale
algorithms reuse the concepts of particle swarm optimization under new names
(2020, 2023) and that cuckoo search brings no new idea to optimization (2022).
Kůdela (2022,
2023) shows that many recent methods contain operators biased towards the
centre of the search space, which flatters them on benchmarks with the optimum
there; nine of the methods here (grey wolf, whale, sine cosine, arithmetic,
Harris hawks, teaching-learning, salp swarm, marine predators, and Jaya) are
among those found biased. Niu, Niu, Liu and Chang (2019) show the defect for the
grey wolf optimizer, Castelli, Manzoni, Mariot, Nobile and Tangherloni (2022)
review the salp swarm algorithm, Črepinšek, Liu and Mernik (2012) correct the
published account of teaching-learning-based optimization, and Gauci, Dodd and
Groß (2012) show that the pull between two members in the gravitational search
algorithm does not depend on their distance, unlike the law of gravity. These
findings concern continuous benchmark functions; how the
biases carry over to the rank-based decode used here has not been measured.
"""

from __future__ import annotations

from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod
from dispatchatlas.solvers.swarm.arithmetic import ArithmeticOptimizationAlgorithm
from dispatchatlas.solvers.swarm.bee_colony import ArtificialBeeColony
from dispatchatlas.solvers.swarm.cuckoo_search import CuckooSearch
from dispatchatlas.solvers.swarm.firefly import FireflyAlgorithm
from dispatchatlas.solvers.swarm.gravitational_search import GravitationalSearchAlgorithm
from dispatchatlas.solvers.swarm.grey_wolf import GreyWolfOptimizer
from dispatchatlas.solvers.swarm.harris_hawks import HarrisHawksOptimization
from dispatchatlas.solvers.swarm.jaya import Jaya
from dispatchatlas.solvers.swarm.marine_predators import MarinePredatorsAlgorithm
from dispatchatlas.solvers.swarm.moth_flame import MothFlameOptimization
from dispatchatlas.solvers.swarm.salp_swarm import SalpSwarmAlgorithm
from dispatchatlas.solvers.swarm.sine_cosine import SineCosineAlgorithm
from dispatchatlas.solvers.swarm.teaching_learning import TeachingLearningBasedOptimization
from dispatchatlas.solvers.swarm.whale import WhaleOptimizationAlgorithm

__all__ = [
    "METHODS",
    "ArithmeticOptimizationAlgorithm",
    "ArtificialBeeColony",
    "CuckooSearch",
    "FireflyAlgorithm",
    "GravitationalSearchAlgorithm",
    "GreyWolfOptimizer",
    "HarrisHawksOptimization",
    "Jaya",
    "MarinePredatorsAlgorithm",
    "MothFlameOptimization",
    "SalpSwarmAlgorithm",
    "SineCosineAlgorithm",
    "Swarm",
    "SwarmMethod",
    "TeachingLearningBasedOptimization",
    "WhaleOptimizationAlgorithm",
    "register",
]

METHODS: tuple[type[SwarmMethod], ...] = (
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
    FireflyAlgorithm,
    MothFlameOptimization,
    SineCosineAlgorithm,
    ArithmeticOptimizationAlgorithm,
    HarrisHawksOptimization,
    TeachingLearningBasedOptimization,
    SalpSwarmAlgorithm,
    MarinePredatorsAlgorithm,
    CuckooSearch,
    ArtificialBeeColony,
    GravitationalSearchAlgorithm,
    Jaya,
)
"""Every method of the package, in the order of the list above."""


def register(registry: SolverRegistry) -> None:
    """Add the population metaheuristics on priority vectors to ``registry``."""
    for method in METHODS:
        registry.register(method)
