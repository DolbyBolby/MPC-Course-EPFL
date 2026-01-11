# Justification des Matrices de Coût Q et R

## Introduction

Les matrices Q et R définissent les pondérations du coût dans le problème d'optimisation MPC. Le choix de ces matrices est crucial pour:
- **Q**: Pénaliser les écarts par rapport à l'équilibre (régulation de l'état)
- **R**: Pénaliser les efforts de contrôle (limitation de l'entrée de commande)

La sélection repose sur l'analyse dynamique, les contraintes physiques et les objectifs de contrôle de chaque sous-système.

---

## 1. Controller XVEL (Contrôle de la Vélocité X)

### Matrices Utilisées
$$Q = \begin{bmatrix} 5 & 0 & 0 \\ 0 & 200 & 0 \\ 0 & 0 & 50 \end{bmatrix}, \quad R = \begin{bmatrix} 1 \end{bmatrix}$$

### États du Système (Indices [1, 4, 6])
- **x₀**: $\omega_y$ - Vitesse angulaire de tangage (rad/s)
- **x₁**: $\beta$ - Angle de tangage (pitch angle) (rad)
- **x₂**: $v_x$ - Vélocité longitudinale X (m/s)

### Entrée de Contrôle (Indice [1])
- **u₀**: $F_y$ - Force de thrust en Y

### Justification

#### Q Matrix

**Q₀₀ = 5 (Vitesse angulaire de tangage ωy)**
- Pénalité **faible** sur la vitesse angulaire
- La dérivée de l'angle est un état secondaire dans ce contrôleur
- Permet une certaine réactivité angulaire

**Q₁₁ = 200 (Angle de tangage β) - ÉLÉMENT CRITIQUE**
- **Pénalité TRÈS ÉLEVÉE** pour forcer un angle de tangage minimal
- **Justification physique directe**: β est l'angle de pitch qui doit respecter les contraintes: $|\beta| \leq 0.1745$ rad (±10°)
- Une pénalité forte force β proche de l'équilibre β_s
- Le thrust Y provoque une rotation de tangage; limiter fortement β évite les violations de contrainte
- Cette pénalité agit comme un **soft constraint** préventif

**Q₂₂ = 50 (Vélocité X vx)**
- Pénalité modérée sur la vélocité X (objectif principal du contrôleur)
- Encourage la convergence vers la vélocité X désirée sans être trop agressif
- Équilibre entre poursuite de référence et contrainte d'angle

#### R Matrix

**R = 1.0 (Effort de Thrust Y)**
- Pénalité modérée sur l'entrée de contrôle
- Limite les variations du thrust Y pour assurer la stabilité
- Empêche les corrections trop agressives qui augmenteraient β

### Résumé xvel
```
Priorité: Angle β << Vélocité vx < Vitesse angulaire ωy
Objectif Principal: Suivre vx en maintenant β ≈ β_s
```
L'accent principal est mis sur **la limitation de l'angle de tangage β** pour respecter les contraintes, suivi du **suivi de la vélocité X vx**.

---

## 2. Controller YVEL (Contrôle de la Vélocité Y)

### Matrices Utilisées
$$Q = \begin{bmatrix} 5 & 0 & 0 \\ 0 & 200 & 0 \\ 0 & 0 & 50 \end{bmatrix}, \quad R = \begin{bmatrix} 1 \end{bmatrix}$$

### États du Système (Indices [0, 3, 7])
- **x₀**: $\omega_x$ - Vitesse angulaire de roulis (rad/s)
- **x₁**: $\alpha$ - Angle de roulis (roll angle) (rad)
- **x₂**: $v_y$ - Vélocité latérale Y (m/s)

### Entrée de Contrôle (Indice [0])
- **u₀**: $F_x$ - Force de thrust en X

### Justification

#### Symétrie Complète avec XVEL

Les matrices Q et R sont **identiques** à celles de xvel pour plusieurs raisons:

1. **Symétrie du système**: Le rocket a une dynamique symétrique selon les axes X et Y
   - Les deux axes utilisent le même type d'actionneur (thrust vectorisé)
   - Les mêmes contraintes d'angle s'appliquent: $|\alpha| \leq 0.1745$ rad (±10°)

2. **Couplage identique**:
   - **xvel**: Thrust Y → Rotation de tangage β
   - **yvel**: Thrust X → Rotation de roulis α
   - Les deux présentent le même défi de contrôle: limiter l'angle pour respecter les contraintes

3. **Même objectif**: Réguler une vélocité latérale (vx ou vy) tout en limitant l'angle correspondant (β ou α)

#### Interprétation des Valeurs

**Q₀₀ = 5**: Vitesse angulaire ωx faiblement pénalisée
**Q₁₁ = 200**: Angle de roulis α fortement limité pour respecter $|\alpha| \leq 10°$
**Q₂₂ = 50**: Vélocité Y vy suivie avec pénalité modérée
**R = 1.0**: Thrust X Fy limité de manière modérée

### Résumé yvel
```
Priorité: Angle α << Vélocité vy < Vitesse angulaire ωx
Objectif Principal: Suivre vy en maintenant α ≈ α_s
Structure identique à xvel par symétrie
```

---

## 3. Controller ZVEL (Contrôle de la Vélocité Z)

### Matrices Utilisées
$$Q = \begin{bmatrix} 50 & 0 \\ 0 & 50 \end{bmatrix}, \quad R = \begin{bmatrix} 0.1 \end{bmatrix}$$

### États du Système (Indices [9, 11])
- **x₀**: $v_z$ - Vélocité verticale Z (m/s)
- **x₁**: $a_z$ - Accélération verticale Z (m/s²)

### Entrée de Contrôle (Indice [2])
- **u₀**: $F_z$ - Force de thrust en Z

### Justification

#### Différences Majeures avec XVEL/YVEL

**1. Pas de Couplage avec les Angles**
- Le thrust Z n'affecte pas directement β ou α
- Pas de contrainte d'angle de pitch/roll à respecter via Z
- Le système est **complètement découplé** des autres axes d'attitude
- **Conséquence**: Pas besoin d'une pénalité très élevée pour limiter un angle

**2. Q₀₀ = Q₁₁ = 50 (Valeurs Modérées et Équilibrées)**
- Pénalité **égale** sur vélocité Z et accélération Z
- Pas de limitation agressive comme dans xvel/yvel (pas de contrainte d'angle)
- Les deux états contribuent **équitablement** au coût
- Permet une convergence rapide et naturelle vers la vélocité Z désirée
- **Comparaison**:
  ```
  xvel/yvel: Q₁₁ = 200 (10x plus grand que xvel/yvel: Q₀₀ = 5)
  zvel:      Q₀₀ = 50, Q₁₁ = 50 (ratio 1:1, équilibré)
  ```

**3. R = 0.1 (TRÈS BAS COMPARÉ À XVEL/YVEL)**
- **Justification critique**: Le rocket a une très **basse inertie en Z**
- **Inertie Z << Inertie X/Y** → Le thrust Z peut être varié rapidement sans instabilité
- R=0.1 (10 fois plus bas que xvel/yvel où R=1.0) autorise des variations agressives du thrust
- Permet d'atteindre rapidement la vélocité Z cible
- **Pas de contrainte physique** limitant le thrust Z comme avec xvel/yvel (pas d'angle à respecter)

#### Interprétation Physique

```
Thrust Z → Accélération Z verticale → Vélocité Z
         ↓
         (Sans contrainte d'angle)
         
Sans contrainte d'angle, le contrôle peut être très agressif.
La basse inertie en Z permet des changements rapides et stables.
```

### Résumé zvel
```
Priorité: Vélocité Z ≈ Accélération Z (équilibrée, pénalités modérées)
Avantage Clé: Pas de contrainte d'angle → libertée de contrôle
Effort: Très agressif (R=0.1) car peu de limitations physiques
```

---

## 4. Controller ROLL (Contrôle de l'Angle de Roulis)

### Matrices Utilisées
$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad R = \begin{bmatrix} 1 \end{bmatrix}$$

### États du Système (Indices [2, 5])
- **x₀**: $\omega_z$ - Vitesse angulaire de lacet (yaw rate) (rad/s)
- **x₁**: $\gamma$ - Angle de lacet (yaw angle) (rad)

### Entrée de Contrôle (Indice [3])
- **u₀**: $\tau_z$ - Moment de contrôle en Z (moment de lacet)

### Justification

#### Pénalités Minimales (Q = I, R = 1)

**1. Q₀₀ = Q₁₁ = 1 (Pénalités Très Faibles)**
- Pénalité **uniforme et très basse** sur vitesse angulaire et angle de lacet
- Le yaw est un angle **directement contrôlé** par le moment de lacet τz
- Pas de découplage avec d'autres dynamiques comme xvel/yvel
- Permet des changements rapides d'angle de lacet si nécessaire

**Contraste Critique avec XVEL/YVEL**:
```
xvel/yvel: Q₁₁ = 200 (limitation TRÈS forte sur angle β/α)
roll:      Q₀₀ = 1,  Q₁₁ = 1  (pénalités TRÈS faibles sur ωz et γ)

Raison clé: 
  xvel/yvel:  Les angles β/α sont des EFFETS SECONDAIRES du thrust latéral
              → Besoin de forte limitation pour respecter |β|, |α| ≤ ±10°
  roll:       L'angle γ est DIRECTEMENT CONTRÔLÉ par le moment τz
              → Pas de contrainte d'angle découplée
              → Contrôle plus doux possible
```

**2. R = 1.0 (Effort Modéré)**
- Équilibré avec les pénalités d'état très faibles
- Évite les corrections trop agressives du moment de lacet
- Maintient la stabilité rotationnelle sans être trop limitant

#### Interprétation Physique

```
Moment de Lacet τz → Accélération Angulaire ωż → Vitesse Angulaire ωz → Angle γ

Dynamique DIRECTE: Pas de contrainte d'angle découplée (comme le pitch limite xvel/yvel).
Le contrôle peut être plus doux et moins réactif car aucune contrainte
physique ne force une limitation aggressive de l'angle de lacet.
```

#### Cas Spécial: Lacet vs Pitch/Roll

- **Pitch/Roll** (β, α): Limités par contraintes aérodynamiques strictes (±10°)
  - → Q très élevé (200) pour forcer l'angle proche de l'équilibre
- **Yaw** (γ): Pas de contrainte aérodynamique stricte
  - → Q très faible (1) pour permettre une plus grande liberté

### Résumé roll
```
Priorité: Vitesse angulaire ωz ≈ Angle γ (équilibrée, pénalités très basses)
Avantage: Contrôle direct et plus doux
Effort: Modéré (R=1.0)
Stratégie: Contrôle bienveillant, peu agressif, contrainte mineure
```

---

## Tableau Comparatif Détaillé

| Aspect | XVEL | YVEL | ZVEL | ROLL |
|:------:|:----:|:----:|:----:|:----:|
| **États** | ωy, β, vx | ωx, α, vy | vz, az | ωz, γ |
| **Q max** | 200 | 200 | 50 | 1 |
| **Q Structure** | [5, 200, 50] | [5, 200, 50] | [50, 50] | [1, 1] |
| **Stratégie Q** | Agressif sur angle | Agressif sur angle | Équilibré | Doux |
| **Raison Q** | Limiter β à ±10° | Limiter α à ±10° | Pas de limite d'angle | Direct, pas de limite |
| **R** | 1.0 | 1.0 | 0.1 | 1.0 |
| **Stratégie R** | Modéré | Modéré | Très agressif | Modéré |
| **Raison R** | Équilibre, stabilitée | Équilibre, stabilité | Basse inertie Z | Stabilité |
| **Couplage Angle** | β ← Fy | α ← Fx | Aucun | Aucun |
| **Contrainte d'Angle** | \|β\| ≤ 10° | \|α\| ≤ 10° | Aucune | Aucune |
| **Découplage** | Indirect (via angle) | Indirect (via angle) | Direct | Direct |

---

## Principes Généraux de Conception

### Principe 1: Limitation des Angles via Pénalité Directe
```
Pour xvel/yvel: L'angle β ou α doit respecter |angle| ≤ 10°
Ces angles sont DIRECTEMENT DANS L'ÉTAT (pas d'effet secondaire)
Donc:  Q_angle = 200 >> Q_velocity pour forcer angle ≈ angle_s
```

### Principe 2: Contrôle Direct vs Contrôle Indirect
```
CONTRÔLE DIRECT:
  - roll (γ directement contrôlé par τz)
  - zvel (vz directement contrôlé par Fz)
  → Pénalités basses (1 ou 50) car peu de contraintes

CONTRÔLE INDIRECT:
  - xvel (vx contrôlé via Fy, mais Fy affecte aussi β)
  - yvel (vy contrôlé via Fx, mais Fx affecte aussi α)
  → Pénalités très fortes sur angle (200) pour maîtriser l'effet secondaire
```

### Principe 3: Limitation par Inertie et Viscosité
```
Basse inertie (Z) → Thrust peut varier rapidement sans instabilité → R faible (0.1)
Inertie standard (X,Y) → Thrust modéré pour stabilité → R fort (1.0)
Inertie rotationnelle → Moment modéré pour stabilité angulaire → R fort (1.0)
```

### Principe 4: Contraintes Physiques Strictes
```
Contrainte stricte sur pitch/roll (±10°) → Pénalités TRÈS fortes (200) sur l'angle
Pas de contrainte sur yaw ou thrust Z → Pénalités basses (1 ou 0.1)
```

---

## Tuning et Optimisation Future

### Si Convergence Trop Lente
- **Réduire Q**: Moins de pénalité sur l'état (convergence rapide mais moins précise)
- **Réduire R**: Autoriser plus d'effort de contrôle (réponse plus agressif)

Exemple pour zvel: R passe de 0.1 à 0.05

### Si Oscillations ou Instabilité
- **Augmenter Q**: Plus de pénalité sur l'état (amortissement)
- **Augmenter R**: Limiter les efforts de contrôle (moins d'agressivité)

Exemple pour xvel: Q₁₁ passe de 200 à 400

### Si Violation de Contraintes
- **Augmenter Q_contrainte**: Augmenter la pénalité sur l'état violant la contrainte

Exemple pour xvel: Q₁₁ = 200 → 300 si pitch angle viole ±10°

---

## Conclusion

Le choix des matrices Q et R reflète une **stratégie d'équilibre** entre:
1. **Faisabilité**: Respecter les contraintes d'angle (xvel/yvel)
2. **Performance**: Converger vers les références de vélocité
3. **Effort**: Limiter l'agressivité du contrôle selon l'inertie du système
4. **Stabilité**: Éviter les oscillations et instabilités

**La hiérarchie de priorités** découle de la **structure physique du rocket** et de ses **contraintes opérationnelles**.
