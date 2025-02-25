
from cc3d import CompuCellSetup
        

from fpp_polarity_force_Steppables import migration_racdir_fppSteppable
CompuCellSetup.register_steppable(steppable=migration_racdir_fppSteppable(frequency=1))

from fpp_polarity_force_Steppables import OdeSteppable
CompuCellSetup.register_steppable(steppable=OdeSteppable(frequency=1))

from fpp_polarity_force_Steppables import FocalPointPlasticityParams
CompuCellSetup.register_steppable(steppable=FocalPointPlasticityParams(frequency=10))

from fpp_polarity_force_Steppables import OdeUpdateParams
CompuCellSetup.register_steppable(steppable=OdeUpdateParams(frequency=1))

CompuCellSetup.run()