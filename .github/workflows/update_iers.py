import os
import subprocess
from batchpr import Updater
from datetime import datetime

# Use a branch name with the date/time to ensure unique branch names each time
# the update is run.
BRANCH_NAME =  'update-iers-tables-' + datetime.now().strftime("%Y%m%d%H%M%S")


class IERSUpdater(Updater):

    def process_repo(self):
        subprocess.call('./update_builtin_iers.sh',
                        cwd='astropy/utils/iers/data')
        self.add('astropy/utils/iers/data/Leap_Second.dat')
        self.add('astropy/utils/iers/data/eopc04_IAU2000.62-now')
        return True

    @property
    def commit_message(self):
        return "Update IERS Earth rotation and leap second tables"

    @property
    def branch_name(self):
        return BRANCH_NAME

    @property
    def pull_request_title(self):
        return self.commit_message

    @property
    def pull_request_body(self):
        return "This is an automated update of the IERS Earth rotation and leap second tables. Feel free to merge if the CI passes!"


helper = IERSUpdater(token=os.environ['GH_TOKEN'],
                     author_name='astrobot',
                     author_email='tom@chi-squared.org')

helper.run('astropy/astropy')
